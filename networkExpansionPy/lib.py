from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
import os
import json
from copy import copy, deepcopy
import zipfile
import pickle
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# Try to import Rust acceleration module
try:
    import netexprs
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

# define asset path
asset_path,filename = os.path.split(os.path.abspath(__file__))
asset_path = asset_path + '/assets'

def load_metabolism(fname):
    return pd.read_pickle(asset_path  + "/metabolic_networks/" + fname)

def load_compounds(fname):
    return pd.read_csv(asset_path  + "/compounds/" + fname)


def netExp(R,P,x,b):
    k = np.sum(x);
    k0 = 0;
    n_reactions = np.size(R,1)
    y = csr_matrix(np.zeros(n_reactions))
    while k > k0:
        k0 = np.sum(x);
        y = (np.dot(R.transpose(),x) == b);
        y = y.astype('int');
        x_n = np.dot(P,y) + x;
        x_n = x_n.astype('bool');
        x = x_n.astype('int');
        k = np.sum(x);
    return x,y

# single step of network expansion
def netExp_step(R,P,x,b):
    k = np.sum(x);
    k0 = 0;
    n_reactions = np.size(R,1)
    y = csr_matrix(np.zeros(n_reactions))

    k0 = np.sum(x);
    y = (np.dot(R.transpose(),x) == b);
    y = y.astype('int');
    x_n = np.dot(P,y) + x;
    x_n = x_n.astype('bool');
    x = x_n.astype('int');
    k = np.sum(x);
    return x,y

# define a new network expansion, s.t. stopping criteria is now no new compounds or reactions can be added at subsequent iterations
def netExp_cr(R,P,x,b):
    k = np.sum(x);
    k0 = 0;
    n_reactions = np.size(R,1)
    y = csr_matrix(np.zeros(n_reactions))
    l = 0
    l0 = 0;
        
    while (k > k0) | (l > l0):
        k0 = np.sum(x);
        l0 = np.sum(y)
        y = (np.dot(R.transpose(),x) == b);
        y = y.astype('int');
        x_n = np.dot(P,y) + x;
        x_n = x_n.astype('bool');
        x = x_n.astype('int');
        k = np.sum(x);
        l = np.sum(y)
    return x,y


def netExp_trace(R,P,x,b):
    
    X = []
    Y = []
    
    X.append(x)
    k = np.sum(x);
    k0 = 0;
    n_reactions = np.size(R,1)
    y = csr_matrix(np.zeros(n_reactions))
    Y.append(y)
    
    while k > k0:
        k0 = np.sum(x);
        y = (np.dot(R.transpose(),x) == b);
        y = y.astype('int');
        x_n = np.dot(P,y) + x;
        x_n = x_n.astype('bool');
        x = x_n.astype('int');
        k = np.sum(x);
        X.append(x)
        Y.append(y) 
    return X,Y



def netContract(R,P,b,x_ac,y,yext):
    # code for running network contraction algorithm
    Xac = []
    Yac = []
    
    #x_ac = x.copy()
    Xac.append(x_ac)
    k0 = np.sum(x_ac);
    n_reactions = np.size(R,1)
    #yactive = csr_matrix(np.multiply((y.toarray()),1-yext.toarray()))   
    yactive = yext.__lt__(1).multiply(y)
    k0 = yactive.sum()
    Yac.append(yactive)
    k = 0
    while k < k0:
        # find
        k0 = yactive.sum();
        x_ex = np.dot(P,yext).astype('bool').astype('int')
        x_ac = np.dot(P,yactive).astype('bool').astype('int')
        # find all extinct metabolites
        #x_ex = csr_matrix(np.multiply((x_ex.toarray()),1-x_ac.toarray()))
        
        # metabolite has to be not active and extinct
        x_ex = x_ac.__lt__(1).multiply(x_ex)
        
        #compute extinct reactions
        yext = np.dot(R.transpose(),x_ex).astype('bool').astype('int')
        # compute active reactions
        
        #yactive = csr_matrix(np.multiply((yactive.toarray()),1-yext.toarray()))   
        yactive = yext.__lt__(1).multiply(yactive)
        #k = np.sum(x_ac);
        k = yactive.sum()
        Xac.append(x_ac)
        Yac.append(yactive)
        
        
    return Xac,Yac


def parse_reaction_trace(reaction_trace,network):
    rxns_list = []
    for i in range(1,len(reaction_trace)):
        idx = reaction_trace[i].nonzero()[0]
        rxns = list(network.iloc[:,idx])
        rxns = pd.DataFrame(rxns,columns = ['rn','direction'])
        rxns['iter'] = i
        rxns_list.append(rxns)    
    rxns_list = pd.concat(rxns_list,axis=0)
    return rxns_list


def isRxnCoenzymeCoupled(rxn,cosubstrate,coproduct):
    g = rxn[rxn.cid.isin([cosubstrate,coproduct])]
    out = False
    if len(g) > 1:
        if g.s.sum() == 0:
            out = True
    return out

def load_ecg_network(ecg):
    network_list = []
    consistent_rids = []
    for rid,v in ecg["reactions"].items():
        cids = v["left"] + v["right"]
        try: ## This skips all reactions with n stoichiometries
            stoichs = [-int(i) for i in v["metadata"]["left_stoichiometries"]]+[int(i) for i in v["metadata"]["right_stoichiometries"]]
            network_list+=list(zip(cids,[rid for _ in range(len(stoichs))],stoichs))
        except:
            pass
        if v["metadata"]["element_conservation"]==True:
            consistent_rids.append(rid)
    return pd.DataFrame(network_list,columns=("cid","rn","s")), pd.DataFrame(consistent_rids,columns=["rn"])

def load_json_network(rdict):
    network_list = []
    consistent_rids = []
    for rid,v in rdict.items():
        if v["glycans"] == False: ## This skips all reactions with glycans
            cids = v["left"] + v["right"]
            try: ## This skips all reactions with n stoichiometries
                stoichs = [-int(i) for i in v["left_stoichiometries"]]+[int(i) for i in v["right_stoichiometries"]]
                network_list+=list(zip(cids,[rid for _ in range(len(stoichs))],stoichs))
            except:
                pass
            if v["element_conservation"]==True:
                consistent_rids.append(rid)
    return pd.DataFrame(network_list,columns=("cid","rn","s")), pd.DataFrame(consistent_rids,columns=["rn"])

def _load_tuple_network(tlist):
    """
    Load a simple network, defined by 2-tuples of reactant lists and product lists
    
    Note: Intended for DEV only
    Note: Stoichiometry information in output is only accurate for directionality
    
        tlist_example = [
             (["A","B"],["C"]),
             (["C","D"],["E","F"]),
             (["E","F"],["G"]),
             (["G","H"],["I"]),
             (["A","J"],["I"])]
    """
    rows = list()
    for i,d in enumerate(tlist):
        if len(d)!=2: raise ValueError("reactions must be two-tuples")
        for cid in d[0]:
            rows.append({"rn":i, "cid":cid, "s":-1})
        for cid in d[1]:
            rows.append({"rn":i, "cid":cid, "s":1})
    return pd.DataFrame(rows)

class GlobalMetabolicNetwork:
    
    def __init__(self,metabolism="KEGG_OG"):
        # load the data        
        if metabolism == "KEGG_OG":
            network = pd.read_csv(asset_path + '/KEGG/network_full.csv')
            cpds = pd.read_csv(asset_path +'/compounds/cpds.txt',sep='\t')
            thermo = pd.read_csv(asset_path +'/reaction_free_energy/kegg_reactions_CC_ph7.0.csv',sep=',')
            self._network = network
            self.thermo = thermo
            self.compounds = cpds ## Includes many compounds without reactions
            
        
        elif metabolism == "ecg":
            with open(os.path.join(asset_path,"ecg","master_from_kegg_2021-01-05.json")) as f:
                ecg = json.load(f)
            network, consistent_rxns = load_ecg_network(ecg)
            self._network = network
            self.consistent_rxns = consistent_rxns
            self.compounds = pd.DataFrame(network["cid"].unique(),columns=["cid"]) ## Only includes compounds with reactions

        elif metabolism == "KEGG":
            with zipfile.ZipFile(os.path.join(asset_path,"KEGG","2021.05.31-18.06.52","entries_detailed","reaction.json.zip"),"r") as z:
                rdict = json.loads(z.read(z.infolist()[0]).decode())
            network, consistent_rxns = load_json_network(rdict)
            self._network = network
            self.consistent_rxns = consistent_rxns
            self.compounds = pd.DataFrame(network["cid"].unique(),columns=["cid"]) ## Only includes compounds with reactions

        elif metabolism == "dev":
            ## Just for testing, etc.
            self._network = None

        else:
            raise(ValueError("'metabolism' must be one of 'KEGG_OG, 'ecg', 'KEGG'"))

        self.metabolism = metabolism
        self.temperature = 25
        self.seedSet = None
        self.rid_to_idx = None
        self.idx_to_rid = None
        self.cid_to_idx = None
        self.idx_to_cid = None
        self.S = None
        self._rust_arrays = None  # Lazily initialized cache for Rust arrays

    def __getstate__(self):
        state = self.__dict__.copy()
        # Ensure we save as _network
        if 'network' in state and '_network' not in state:
            state['_network'] = state.pop('network')
        return state

    def __setstate__(self, state):
        # Handle old pickles that have 'network' instead of '_network'
        if 'network' in state and '_network' not in state:
            state['_network'] = state.pop('network')
        # Handle old pickles that don't have '_rust_arrays'
        if '_rust_arrays' not in state:
            state['_rust_arrays'] = None
        self.__dict__.update(state)

    @property
    def network(self):
        return self._network

    @network.setter
    def network(self, value):
        self._network = value
        self._invalidate_rust_cache()
        
    def _ensure_dicts(self):
        """Ensure compound and reaction dictionaries are initialized."""
        if self.rid_to_idx is None or self.idx_to_rid is None:
            self.rid_to_idx, self.idx_to_rid = self.create_reaction_dicts()
        if self.cid_to_idx is None or self.idx_to_cid is None:
            self.cid_to_idx, self.idx_to_cid = self.create_compound_dicts()
    
    def _ensure_rust_ready(self):
        """Lazily prepare arrays needed for Rust acceleration."""
        if self._rust_arrays is not None:
            return
        
        self._ensure_dicts()
        
        # Build R and P matrices
        R, P = self.create_RP_from_irreversible_network()
        R = csr_matrix(R)
        P = csr_matrix(P)
        
        # Compute b vector
        b = np.asarray(R.sum(axis=0)).ravel()
        
        # Convert R^T to CSR format for Rust
        R_T = R.T.tocsr()
        
        # Cache all arrays needed by Rust functions
        self._rust_arrays = {
            'rt_data': R_T.data.astype(np.float64),
            'rt_indices': R_T.indices.astype(np.int32),
            'rt_indptr': R_T.indptr.astype(np.int32),
            'p_data': P.data.astype(np.float64),
            'p_indices': P.indices.astype(np.int32),
            'p_indptr': P.indptr.astype(np.int32),
            'n_reactions': R.shape[1],
            'n_compounds': P.shape[0],
            'b': b.astype(np.float64),
            # Also cache sparse matrices for Python fallback
            'R': R,
            'P': P,
            'b_sparse': csr_matrix(b).T,
        }
    
    def _invalidate_rust_cache(self):
        """Invalidate cached Rust arrays. Call this after modifying the network."""
        self._rust_arrays = None
        self.rid_to_idx = None
        self.idx_to_rid = None
        self.cid_to_idx = None
        self.idx_to_cid = None
    
    def _x_to_compounds(self, x_arr):
        """Convert compound boolean array to list of compound IDs."""
        if isinstance(x_arr, np.ndarray):
            cidx = np.nonzero(x_arr)[0]
        else:
            # Handle sparse matrix
            cidx = np.nonzero(x_arr.toarray().ravel())[0]
        return [self.idx_to_cid[i] for i in cidx]
    
    def _y_to_reactions(self, y_arr):
        """Convert reaction boolean array to list of reaction IDs."""
        if isinstance(y_arr, np.ndarray):
            ridx = np.nonzero(y_arr)[0]
        else:
            # Handle sparse matrix
            ridx = np.nonzero(y_arr.toarray().ravel())[0]
        return [self.idx_to_rid[i] for i in ridx]

    def copy(self):
        return deepcopy(self)
        
    def set_ph(self,pH):
        if ~(type(pH) == str):
            pH = str(pH)
        if self.metabolism == "KEGG_OG":
            try:
                thermo = pd.read_csv(asset_path + '/reaction_free_energy/kegg_reactions_CC_ph' + pH + '.csv',sep=',')
                self.thermo = thermo
            except Exception as error:
                print('Failed to open pH files (please use 5.0-9.0 in 0.5 increments)')    
        elif self.metabolism == "ecg":
            try:
                self.thermo = self.load_ecg_thermo(self.metabolism,pH)
            except:
                raise ValueError("Try another pH, that one appears not to be in the ecg json")
        else:
            raise(NotImplementedError("pH not yet implemented for metabolism = %s"%self.metabolism)) 


    def load_ecg_thermo(self,ph=9):
        thermo_list = []
        for rid,v in self.metabolism["reactions"].items():
            
            phkey = str(ph)+"pH_100mM"
            
            if v["metadata"]["dg"][phkey]["standard_dg_prime_value"] == None:
                dg = np.nan
            else:
                dg = v["metadata"]["dg"][phkey]["standard_dg_prime_value"]
                
            if v["metadata"]["dg"][phkey]["standard_dg_prime_error"] == None:
                dgerror = np.nan
            else:
                dgerror = v["metadata"]["dg"][phkey]["standard_dg_prime_error"]
                
            if v["metadata"]["dg"][phkey]["is_uncertain"] == None:
                note = "uncertainty is too high"
            else:
                note = np.nan

            thermo_list.append((rid,
                dg,
                dgerror,
                v["metadata"]["dg"][phkey]["p_h"],
                v["metadata"]["dg"][phkey]["ionic_strength"]/1000,
                v["metadata"]["dg"][phkey]["temperature"],
                note)) 

        return pd.DataFrame(thermo_list, columns = ("!MiriamID::urn:miriam:kegg.reaction","!dG0_prime (kJ/mol)","!sigma[dG0] (kJ/mol)","!pH","!I (mM)","!T (Kelvin)","!Note"))         

    
    def pruneInconsistentReactions(self):
        # remove reactions with qualitatively different sets of elements in reactions and products
        if self.metabolism=="KEGG_OG":
            consistent = pd.read_csv(asset_path + '/reaction_sets/reactions_consistent.csv')
            self.network = self.network[self.network.rn.isin(consistent.rn.tolist())]
        elif self.metabolism=="KEGG" or self.metabolism=="ecg":
            self.network = self.network[self.network.rn.isin(self.consistent_rxns.rn.tolist())]
        else:
            raise(NotImplementedError("Function not yet implemented for metabolism = %s"%self.metabolism)) 
        self._invalidate_rust_cache()

    def pruneUnbalancedReactions(self):
        # only keep reactions that are elementally balanced
        if self.metabolism=="KEGG_OG":
            balanced = pd.read_csv(asset_path + '/reaction_sets/reactions_balanced.csv')
            self.network = self.network[self.network.rn.isin(balanced.rn.tolist())]
        else:
            raise(NotImplementedError("Function not yet implemented for metabolism = %s"%self.metabolism)) 
        self._invalidate_rust_cache()
        
    def subnetwork(self,rxns):
        # only keep reactions that are in list
        self.network = self.network[self.network.rn.isin(rxns)]
        self._invalidate_rust_cache()
        
    def addGenericCoenzymes(self):
        replace_metabolites = {'C00003': 'Generic_oxidant', 'C00004': 'Generic_reductant', 'C00006': 'Generic_oxidant',  'C00005': 'Generic_reductant','C00016': 'Generic_oxidant','C01352':'Generic_reductant'}
        coenzyme_pairs = {}
        coenzyme_pairs['NAD'] = ['C00003','C00004']
        coenzyme_pairs['NADP'] = ['C00006','C00005']
        coenzyme_pairs['FAD'] = ['C00016','C01352']
        coenzyme_pairs = pd.DataFrame(coenzyme_pairs).T.reset_index()
        coenzyme_pairs.columns = ['id','oxidant','reductant']
        # create reactions copies with coenzyme pairs
        new_rxns = []
        new_thermo = [];
        for idx,rxn in self.network.groupby('rn'):
            z = any([isRxnCoenzymeCoupled(rxn,row.oxidant,row.reductant) for x,row in coenzyme_pairs.iterrows()])
            if z:
                new_rxn = rxn.replace(replace_metabolites).groupby(['cid','rn']).sum().reset_index()
                new_rxn['rn'] = new_rxn['rn'] = idx + '_G'
                new_rxns.append(new_rxn)
                t = self.thermo[self.thermo['!MiriamID::urn:miriam:kegg.reaction'] == idx].replace({idx:  idx + '_G'})
                new_thermo.append(t)

        new_rxns = pd.concat(new_rxns,axis=0)
        new_thermo = pd.concat(new_thermo,axis=0)

        self.network = pd.concat([self.network,new_rxns],axis=0)
        self.thermo = pd.concat([self.thermo,new_thermo],axis=0)
        self._invalidate_rust_cache()

    
    def convertToIrreversible(self):
        nf = self.network.copy()
        nb = self.network.copy()
        nf['direction'] = 'forward'
        nb['direction'] = 'reverse'
        nb['s'] = -nb['s']
        net = pd.concat([nf,nb],axis=0)
        net = net.set_index(['cid','rn','direction']).reset_index()
        self.network = net
        self._invalidate_rust_cache()
    
    def setMetaboliteBounds(self,ub = 1e-1,lb = 1e-6): 
        
        self.network['ub'] = ub
        self.network['lb'] = lb
      
    def pruneThermodynamicallyInfeasibleReactions(self,keepnan = False):
        fixed_mets = ['C00001','C00080']

        if not hasattr(self, 'thermo'):
            raise(AttributeError("Metabolism has no thermo data."))

        RT = 0.008309424 * (273.15+self.temperature)
        rns  = []
        dirs = []
        dgs = []
        for (rn,direction), dff in self.network.groupby(['rn','direction']):
            effective_deltaG = np.nan
            if rn in self.thermo['!MiriamID::urn:miriam:kegg.reaction'].tolist():
                deltaG = self.thermo[self.thermo['!MiriamID::urn:miriam:kegg.reaction'] == rn]['!dG0_prime (kJ/mol)'].values[0]
                if direction == 'reverse':
                    deltaG = -1*deltaG

                dff = dff[~dff['cid'].isin(fixed_mets)]
                subs = dff[dff['s'] < 0]
                prods = dff[dff['s'] > 0];
                k = np.dot(subs['ub'].apply(np.log),subs['s']) + np.dot(prods['lb'].apply(np.log),prods['s'])

                effective_deltaG = RT*k + deltaG

            dgs.append(effective_deltaG)
            dirs.append(direction)
            rns.append(rn)

        res = pd.DataFrame({'rn':rns,'direction':dirs,'effDeltaG':dgs})
        if keepnan:
            # change effective free energy to negative number, so it passes the next filter
            res['effDeltaG'] = res['effDeltaG'].fillna(-1)
        else:
            res = res.dropna()

        #res = res[res['effDeltaG'] < 0].set_index(['rn','direction'])
        res = res[~(res['effDeltaG'] > 0)].set_index(['rn','direction'])
        res = res.drop('effDeltaG',axis=1)
        self.network = res.join(self.network.set_index(['rn','direction'])).reset_index()
        self._invalidate_rust_cache()
    
    def pruneReactionsFromMetabolite(self,cpds):
        # find all reactions that use metabolites in cpds list
        reactions_to_drop = self.network[self.network.cid.isin(cpds)].rn.unique().tolist()
        reactions_to_keep = [x for x in self.network.rn.unique().tolist() if x not in reactions_to_drop]
        # only keep reactions that do not use that metabolite
        self.subnetwork(reactions_to_keep)

    def initialize_metabolite_vector(self,seedSet):
        if seedSet is None:
            print('No seed set')
        else:
            x0 = np.zeros([len(self.cid_to_idx)],dtype=int)
            for x in set(seedSet)&set(self.cid_to_idx.keys()):
            #for x in set(map(tuple, seedSet)) & set(map(tuple, self.cid_to_idx.keys())):
                x0[self.cid_to_idx[x]] = 1     
            return x0

    def initialize_reaction_vector(self,reactionSet):
        if reactionSet is None:
            print('No reactions in set')
        else:
            x0 = np.zeros([len(self.rid_to_idx)],dtype=int)
            for x in set(reactionSet)&set(self.rid_to_idx.keys()):
                x0[self.rid_to_idx[x]] = 1     
            return x0


    def create_reaction_dicts(self):
        rids = set(zip(self.network["rn"],self.network["direction"]))
        rid_to_idx = dict()
        idx_to_rid = dict()
        for v, k in enumerate(rids):
            rid_to_idx[k] = v
            idx_to_rid[v] = k
        
        return rid_to_idx, idx_to_rid

    def create_compound_dicts(self):
        cids = set(self.network["cid"])
        cid_to_idx = dict()
        idx_to_cid = dict()
        for v, k in enumerate(cids):
            cid_to_idx[k] = v
            idx_to_cid[v] = k
        
        return cid_to_idx, idx_to_cid

    def create_S_from_irreversible_network(self):
        
        S = np.zeros([len(self.cid_to_idx),len(self.rid_to_idx)])
            
        for c,r,d,s in zip(self.network["cid"],self.network["rn"],self.network["direction"],self.network["s"]):
            S[self.cid_to_idx[c],self.rid_to_idx[(r,d)]] = s

        return S


    def create_RP_from_irreversible_network(self):
        
        # only use entries in teh network dataframe with s < 0:
        reactant_network = self.network[self.network.s<0]

        R = np.zeros([len(self.cid_to_idx),len(self.rid_to_idx)])
        
        for c,r,d,s in zip(reactant_network["cid"],reactant_network["rn"],reactant_network["direction"],reactant_network["s"]):
            R[self.cid_to_idx[c],self.rid_to_idx[(r,d)]] = 1

        # only use entries in teh network dataframe with s > 0:
        product_network = self.network[self.network.s>0]

        P = np.zeros([len(self.cid_to_idx),len(self.rid_to_idx)])
        
        for c,r,d,s in zip(product_network["cid"],product_network["rn"],product_network["direction"],product_network["s"]):
            P[self.cid_to_idx[c],self.rid_to_idx[(r,d)]] = 1

        return R,P

    def create_iteration_dict(self,M,idx_to_id):
        idx_iter = dict()
        for i,row in enumerate(M):
            idxs = np.nonzero(row.toarray().T[0])[0]
            for idx in idxs:
                if idx not in idx_iter:
                    idx_iter[idx] = i

        id_iter = dict()
        for idx,i in idx_iter.items():
            id_iter[idx_to_id[idx]] = i  

        return id_iter
        
    def expand(self, seedSet, algorithm='naive', reaction_mask=None):
        """
        Run network expansion from a seed set of compounds.
        
        Args:
            seedSet: List of compound IDs to start expansion from
            algorithm: 'naive', 'cr', 'trace', or 'step'
            reaction_mask: Optional list of reaction IDs to exclude (remove from network)
        
        Returns:
            For 'naive', 'cr', 'step': (compounds, reactions) - lists of IDs in scope
            For 'trace': (compound_dict, reaction_dict) - dicts mapping ID to iteration
        """
        # Use Rust for naive algorithm without trace
        if _HAS_RUST and algorithm.lower() == 'naive':
            return self._expand_rust(seedSet, reaction_mask)
        else:
            return self._expand_python(seedSet, algorithm, reaction_mask)
    
    def _expand_rust(self, seedSet, reaction_mask=None):
        """Rust-accelerated expansion (naive algorithm only).
        
        Args:
            seedSet: List of compound IDs to start expansion from
            reaction_mask: Optional list of reaction IDs to EXCLUDE (remove from network)
        """
        self._ensure_rust_ready()
        ra = self._rust_arrays
        
        # Convert seedSet to x0 array
        x0 = self.initialize_metabolite_vector(seedSet).astype(np.uint8)
        
        if reaction_mask is not None and len(reaction_mask) > 0:
            # Build mask array: 1 for allowed reactions, 0 for excluded
            # reaction_mask contains reactions to EXCLUDE
            mask = np.ones(ra['n_reactions'], dtype=np.uint8)
            for rid in reaction_mask:
                if rid in self.rid_to_idx:
                    mask[self.rid_to_idx[rid]] = 0
            
            x_arr, y_arr = netexprs.expand_masked(
                ra['rt_data'], ra['rt_indices'], ra['rt_indptr'], ra['n_reactions'],
                ra['p_data'], ra['p_indices'], ra['p_indptr'], ra['n_compounds'],
                x0, ra['b'], mask
            )
        else:
            x_arr, y_arr = netexprs.expand(
                ra['rt_data'], ra['rt_indices'], ra['rt_indptr'], ra['n_reactions'],
                ra['p_data'], ra['p_indices'], ra['p_indptr'], ra['n_compounds'],
                x0, ra['b']
            )
        
        compounds = self._x_to_compounds(x_arr)
        reactions = self._y_to_reactions(y_arr)
        return compounds, reactions
    
    def _expand_python(self, seedSet, algorithm='naive', reaction_mask=None):
        """Pure Python expansion (supports all algorithms).
        
        Args:
            seedSet: List of compound IDs to start expansion from
            algorithm: 'naive', 'cr', 'trace', or 'step'
            reaction_mask: Optional list of reaction IDs to EXCLUDE (remove from network)
        """
        self._ensure_dicts()
        
        x0 = self.initialize_metabolite_vector(seedSet)
        R, P = self.create_RP_from_irreversible_network()
        b = sum(R)

        # sparsefy data
        R = csr_matrix(R)
        P = csr_matrix(P)
        b = csr_matrix(b)
        b = b.transpose()

        # Mask out excluded reactions (reaction_mask contains reactions to EXCLUDE)
        if reaction_mask is not None and len(reaction_mask) > 0:
            # Create vector with 1s for reactions to EXCLUDE
            exclude_vec = self.initialize_reaction_vector(reaction_mask)
            # Invert to get 1s for reactions to KEEP
            keep_vec = 1 - exclude_vec
            reaction_mask_mat = csr_matrix(np.diag(keep_vec))
            P = P * reaction_mask_mat
            R = R * reaction_mask_mat

        x0 = csr_matrix(x0)
        x0 = x0.transpose()
        
        if algorithm.lower() == 'naive':
            x, y = netExp(R, P, x0, b)
        elif algorithm.lower() == 'cr':
            x, y = netExp_cr(R, P, x0, b)
        elif algorithm.lower() == 'trace':
            X, Y = netExp_trace(R, P, x0, b)
        elif algorithm.lower() == 'step':
            x, y = netExp_step(R, P, x0, b)
        else:
            raise ValueError('algorithm needs to be naive (compound stopping criteria) or cr (reaction/compound stopping criteria)')
        
        if algorithm.lower() == 'trace':
            compound_iteration_dict = self.create_iteration_dict(X, self.idx_to_cid)
            reaction_iteration_dict = self.create_iteration_dict(Y, self.idx_to_rid)
            return compound_iteration_dict, reaction_iteration_dict
        else:
            compounds = self._x_to_compounds(x)
            reactions = self._y_to_reactions(y)
            return compounds, reactions


    def contract(self, seedSet, reactionScope, compoundScope, extinctReactions):
        """
        Run network contraction starting from an expanded scope.
        
        Args:
            seedSet: Original seed set (unused in contraction, kept for API compatibility)
            reactionScope: List of reaction IDs in the current scope
            compoundScope: List of compound IDs in the current scope
            extinctReactions: List of reaction IDs to remove (extinct)
        
        Returns:
            (compounds, reactions) - lists of IDs remaining after contraction
        """
        if _HAS_RUST:
            return self._contract_rust(reactionScope, compoundScope, extinctReactions)
        else:
            return self._contract_python(reactionScope, compoundScope, extinctReactions)
    
    def _contract_rust(self, reactionScope, compoundScope, extinctReactions):
        """Rust-accelerated contraction."""
        self._ensure_rust_ready()
        ra = self._rust_arrays
        
        # Convert ID lists to arrays
        x_active = self.initialize_metabolite_vector(compoundScope).astype(np.uint8)
        y_active = self.initialize_reaction_vector(reactionScope).astype(np.uint8)
        y_extinct = self.initialize_reaction_vector(extinctReactions).astype(np.uint8)
        
        x_arr, y_arr = netexprs.contract(
            ra['rt_data'], ra['rt_indices'], ra['rt_indptr'], ra['n_reactions'],
            ra['p_data'], ra['p_indices'], ra['p_indptr'], ra['n_compounds'],
            x_active, y_active, y_extinct
        )
        
        compounds = self._x_to_compounds(x_arr)
        reactions = self._y_to_reactions(y_arr)
        return compounds, reactions
    
    def _contract_python(self, reactionScope, compoundScope, extinctReactions):
        """Pure Python contraction."""
        self._ensure_dicts()
        
        # create vectors for reactionScope, compoundScope
        xactive = self.initialize_metabolite_vector(compoundScope)
        yactive = self.initialize_reaction_vector(reactionScope)
        yextinct = self.initialize_reaction_vector(extinctReactions)
        
        R, P = self.create_RP_from_irreversible_network()
        b = sum(R)

        # sparsefy data
        R = csr_matrix(R)
        P = csr_matrix(P)
        b = csr_matrix(b)
        b = b.transpose()

        xactive = csr_matrix(xactive).transpose()
        yactive = csr_matrix(yactive).transpose()
        yextinct = csr_matrix(yextinct).transpose()
        
        # run contraction algorithm
        X, Y = netContract(R, P, b, xactive, yactive, yextinct)
        x = X[-1]
        y = Y[-1]

        compounds = self._x_to_compounds(x)
        reactions = self._y_to_reactions(y)
        return compounds, reactions

    def run_expansions(self, seedSets, algorithm='naive'):
        """
        Run expansion for multiple seed sets.
        
        Args:
            seedSets: List of seed sets (each is a list of compound IDs)
            algorithm: 'naive' or 'trace'
        
        Returns:
            (compoundScopes, reactionScopes) - lists of results for each seed set
        """
        # For naive algorithm with Rust, we can use the optimized single expansion
        # A future optimization could batch these in Rust
        if _HAS_RUST and algorithm.lower() == 'naive':
            self._ensure_rust_ready()
            compoundScopes = []
            reactionScopes = []
            for seedSet in seedSets:
                compounds, reactions = self._expand_rust(seedSet, reaction_mask=None)
                compoundScopes.append(compounds)
                reactionScopes.append(reactions)
            return compoundScopes, reactionScopes
        else:
            return self._run_expansions_python(seedSets, algorithm)
    
    def _run_expansions_python(self, seedSets, algorithm='naive'):
        """Pure Python batch expansion."""
        self._ensure_dicts()
        
        R, P = self.create_RP_from_irreversible_network()
        b = sum(R)

        # sparsefy data
        R = csr_matrix(R)
        P = csr_matrix(P)
        b = csr_matrix(b)
        b = b.transpose()

        compoundScopes = []
        reactionScopes = []
        for seedSet in seedSets:
            x0 = self.initialize_metabolite_vector(seedSet)
            x0 = csr_matrix(x0)
            x0 = x0.transpose()
            
            if algorithm.lower() == 'naive':
                x, y = netExp(R, P, x0, b)
                compounds = self._x_to_compounds(x)
                reactions = self._y_to_reactions(y)
            elif algorithm.lower() == 'trace':
                X, Y = netExp_trace(R, P, x0, b)
                compounds = self.create_iteration_dict(X, self.idx_to_cid)
                reactions = self.create_iteration_dict(Y, self.idx_to_rid)
            else:
                raise ValueError('please define algorithm (trace or naive)')

            compoundScopes.append(compounds)
            reactionScopes.append(reactions)
        return compoundScopes, reactionScopes

    def run_contractions(self, seedSet, reactionScope, compoundScope, extinctReactionSets):
        """
        Run contraction for multiple extinction sets.
        
        Args:
            seedSet: Original seed set (unused, kept for API compatibility)
            reactionScope: List of reaction IDs in the current scope
            compoundScope: List of compound IDs in the current scope
            extinctReactionSets: List of extinction sets (each is a list of reaction IDs to remove)
        
        Returns:
            (compoundScopes, reactionScopes) - lists of results for each extinction set
        """
        if _HAS_RUST:
            return self._run_contractions_rust(reactionScope, compoundScope, extinctReactionSets)
        else:
            return self._run_contractions_python(reactionScope, compoundScope, extinctReactionSets)
    
    def _run_contractions_rust(self, reactionScope, compoundScope, extinctReactionSets):
        """Rust-accelerated batch contraction."""
        self._ensure_rust_ready()
        ra = self._rust_arrays
        
        # Convert scope to arrays (shared across all contractions)
        x_active = self.initialize_metabolite_vector(compoundScope).astype(np.uint8)
        y_active = self.initialize_reaction_vector(reactionScope).astype(np.uint8)
        
        # Build batch extinction matrix
        n_batches = len(extinctReactionSets)
        y_extinct_batch = np.zeros((n_batches, ra['n_reactions']), dtype=np.uint8)
        for i, extinctReactions in enumerate(extinctReactionSets):
            for rid in extinctReactions:
                if rid in self.rid_to_idx:
                    y_extinct_batch[i, self.rid_to_idx[rid]] = 1
        
        # Single Rust call for all contractions
        x_batch, y_batch = netexprs.contract_batch(
            ra['rt_data'], ra['rt_indices'], ra['rt_indptr'], ra['n_reactions'],
            ra['p_data'], ra['p_indices'], ra['p_indptr'], ra['n_compounds'],
            x_active, y_active, y_extinct_batch
        )
        
        # Convert outputs
        compoundScopes = []
        reactionScopes = []
        for i in range(n_batches):
            compounds = self._x_to_compounds(x_batch[i])
            reactions = self._y_to_reactions(y_batch[i])
            compoundScopes.append(compounds)
            reactionScopes.append(reactions)
        
        return compoundScopes, reactionScopes
    
    def _run_contractions_python(self, reactionScope, compoundScope, extinctReactionSets):
        """Pure Python batch contraction."""
        self._ensure_dicts()
        
        # create vectors for reactionScope, compoundScope
        xactive = self.initialize_metabolite_vector(compoundScope)
        yactive = self.initialize_reaction_vector(reactionScope)
        
        R, P = self.create_RP_from_irreversible_network()
        b = sum(R)

        # sparsefy data
        R = csr_matrix(R)
        P = csr_matrix(P)
        b = csr_matrix(b)
        b = b.transpose()

        xactive = csr_matrix(xactive).transpose()
        yactive = csr_matrix(yactive).transpose()

        compoundScopes = []
        reactionScopes = []

        for extinctReactions in extinctReactionSets:
            yextinct = self.initialize_reaction_vector(extinctReactions)
            yextinct = csr_matrix(yextinct).transpose()
            # run contraction algorithm
            X, Y = netContract(R, P, b, xactive, yactive, yextinct)
            x = X[-1]
            y = Y[-1]
            
            compounds = self._x_to_compounds(x)
            reactions = self._y_to_reactions(y)
            compoundScopes.append(compounds)
            reactionScopes.append(reactions)
        
        return compoundScopes, reactionScopes


    def run_expansions_reactionMasks(self, seedSet, maskedReactionSets):
        """
        Run masked expansions (reactions removed from network).
        
        Args:
            seedSet: List of compound IDs to start expansion from
            maskedReactionSets: List of reaction sets to REMOVE (each is a list of reaction IDs)
        
        Returns:
            (compoundScopes, reactionScopes) - lists of results for each mask
        """
        if _HAS_RUST:
            return self._run_expansions_reactionMasks_rust(seedSet, maskedReactionSets)
        else:
            return self._run_expansions_reactionMasks_python(seedSet, maskedReactionSets)
    
    def _run_expansions_reactionMasks_rust(self, seedSet, maskedReactionSets):
        """Rust-accelerated batch masked expansion."""
        self._ensure_rust_ready()
        ra = self._rust_arrays
        
        # Convert seedSet to x0 array (shared across all expansions)
        x0 = self.initialize_metabolite_vector(seedSet).astype(np.uint8)
        
        # Build batch mask matrix
        # maskedReactionSets contains reactions to REMOVE, so mask = 1 - removed
        n_masks = len(maskedReactionSets)
        masks = np.ones((n_masks, ra['n_reactions']), dtype=np.uint8)
        for i, rxns_removed in enumerate(maskedReactionSets):
            for rid in rxns_removed:
                if rid in self.rid_to_idx:
                    masks[i, self.rid_to_idx[rid]] = 0
        
        # Single Rust call for all expansions
        x_batch, y_batch = netexprs.expand_masked_batch(
            ra['rt_data'], ra['rt_indices'], ra['rt_indptr'], ra['n_reactions'],
            ra['p_data'], ra['p_indices'], ra['p_indptr'], ra['n_compounds'],
            x0, ra['b'], masks
        )
        
        # Convert outputs
        compoundScopes = []
        reactionScopes = []
        for i in range(n_masks):
            compounds = self._x_to_compounds(x_batch[i])
            reactions = self._y_to_reactions(y_batch[i])
            compoundScopes.append(compounds)
            reactionScopes.append(reactions)
        
        return compoundScopes, reactionScopes
    
    def _run_expansions_reactionMasks_python(self, seedSet, maskedReactionSets):
        """Pure Python batch masked expansion."""
        self._ensure_dicts()
        
        R, P = self.create_RP_from_irreversible_network()
        b = sum(R)
        # sparsefy data
        R = csr_matrix(R)
        P = csr_matrix(P)
        b = csr_matrix(b)
        b = b.transpose()

        x0 = self.initialize_metabolite_vector(seedSet)
        x0 = csr_matrix(x0)
        x0 = x0.transpose()

        compoundScopes = []
        reactionScopes = []

        for rxns_removed in maskedReactionSets:
            # build new R and P matrices with Masks
            yextinct = self.initialize_reaction_vector(rxns_removed)
            reaction_mask = csr_matrix(np.diag(1 - yextinct))
            Pstar = P * reaction_mask
            Rstar = R * reaction_mask
            # run expansion algorithm
            x, y = netExp(Rstar, Pstar, x0, b)

            compounds = self._x_to_compounds(x)
            reactions = self._y_to_reactions(y)
            compoundScopes.append(compounds)
            reactionScopes.append(reactions)
        
        return compoundScopes, reactionScopes


    def ne_output_to_graph(self,cpds,rxns):
        # build a network constructing metabolites from prior iteration to connecting subsequent iteration
        # input: cpds (rxns) is a dict with compound id (reaction id) as the key, and iteration as the value
        # output: dataframe with target and source node and iteration for  

        graph = {'target': [], 'source': [], 'iteration':[]}
        maxIter = np.array(list(cpds.values())).max()
        
        cpd_iter_df = pd.DataFrame({'cid': [x[0] for x in cpds.items()],'iter': [x[1] for x in cpds.items()]})

        for i in range(maxIter,0,-1):
            molecules = [x[0] for x in cpds.items() if x[1] == i]
            reactions = [x[0] for x in rxns.items() if x[1] == i]
            reactions = self.network.set_index(['rn','direction']).loc[reactions]
            for molecule in molecules:
                # reactions
                # find any reactions that produce this metabolite
                r_sample = reactions[ ( reactions.cid == molecule) & (reactions.s>0)].sample(1)
                # find metabolite in reaction that was produced in previous iteration
                r_sample_cpds = reactions.loc[r_sample.index].set_index('cid').join(cpd_iter_df.set_index('cid'))
                r_sample_cpds = r_sample_cpds[r_sample_cpds.iter == i-1].sample(1)
                molecules_origin = r_sample_cpds.index.tolist()[0]
                graph['target'].append(molecule)
                graph['source'].append(molecules_origin)
                graph['iteration'].append(i)

        graph = pd.DataFrame(graph)
        return graph

    def save(self,name):
        path_to_save = asset_path + '/metabolic_networks/' + name + ".pkl"
        with open(path_to_save, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def rxns2tuple(self,rn_list):
        t = self.network[self.network.rn.isin(rn_list)][['rn','direction']].drop_duplicates()
        rn_list_tuple = list(zip(t.rn.tolist(),t.direction.tolist()))
        return rn_list_tuple


    def run_expansions_parallel(self, seedSets, algorithm='naive'):
        """
        Run expansion for multiple seed sets in parallel.
        
        Note: When Rust acceleration is available, this delegates to run_expansions()
        which uses Rust's native parallelization (Rayon). The _parallel suffix is kept
        for API compatibility.
        
        Args:
            seedSets: List of seed sets (each is a list of compound IDs)
            algorithm: 'naive' or 'trace'
        
        Returns:
            (compoundScopes, reactionScopes) - lists of results for each seed set
        """
        if _HAS_RUST and algorithm.lower() == 'naive':
            # Rust handles parallelization internally
            return self.run_expansions(seedSets, algorithm)
        else:
            # Fall back to Python multiprocessing
            return self._run_expansions_parallel_python(seedSets, algorithm)
    
    def _run_expansions_parallel_python(self, seedSets, algorithm='naive'):
        """Python multiprocessing-based parallel expansion."""
        self._ensure_dicts()
        
        R, P = self.create_RP_from_irreversible_network()
        b = np.sum(R, axis=0)

        R = csr_matrix(R)
        P = csr_matrix(P)
        b = csr_matrix(b)
        b = b.transpose()

        compoundScopes = []
        reactionScopes = []

        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            results = executor.map(expansion_helper, [(self, seedSet, algorithm, R, P, b) for seedSet in seedSets])

        for compounds, reactions in results:
            compoundScopes.append(compounds)
            reactionScopes.append(reactions)

        return compoundScopes, reactionScopes


    def run_expansions_reactionMasks_parallel(self, seedSet, maskedReactionSets):
        """
        Run masked expansions in parallel.
        
        Note: When Rust acceleration is available, this delegates to run_expansions_reactionMasks()
        which uses Rust's native parallelization (Rayon). The _parallel suffix is kept
        for API compatibility.
        
        Args:
            seedSet: List of compound IDs to start expansion from
            maskedReactionSets: List of reaction sets to REMOVE
        
        Returns:
            (compoundScopes, reactionScopes) - lists of results for each mask
        """
        if _HAS_RUST:
            # Rust handles parallelization internally
            return self.run_expansions_reactionMasks(seedSet, maskedReactionSets)
        else:
            # Fall back to Python multiprocessing
            return self._run_expansions_reactionMasks_parallel_python(seedSet, maskedReactionSets)
    
    def _run_expansions_reactionMasks_parallel_python(self, seedSet, maskedReactionSets):
        """Python multiprocessing-based parallel masked expansion."""
        self._ensure_dicts()
        
        R, P = self.create_RP_from_irreversible_network()
        b = np.sum(R, axis=0)

        R = csr_matrix(R)
        P = csr_matrix(P)
        b = csr_matrix(b)
        b = b.transpose()

        compoundScopes = []
        reactionScopes = []

        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            results = executor.map(expansion_helper_reaction_masks, [(self, seedSet, rxns_removed, R, P, b) for rxns_removed in maskedReactionSets])

        for compounds, reactions in results:
            compoundScopes.append(compounds)
            reactionScopes.append(reactions)

        return compoundScopes, reactionScopes


# define an example helper function for parrellel execution of netowrk expansion code
def expansion_helper(args):
    instance, seedSet, algorithm, R, P, b = args
    x0 = instance.initialize_metabolite_vector(seedSet)
    x0 = csr_matrix(x0)
    x0 = x0.transpose()

    if algorithm.lower() == 'naive':
        x, y = netExp(R, P, x0, b)
        if x.toarray().sum() > 0:
            cidx = np.nonzero(x.toarray().T[0])[0]
            compounds = [instance.idx_to_cid[i] for i in cidx]
        else:
            compounds = []

        if y.toarray().sum() > 0:
            ridx = np.nonzero(y.toarray().T[0])[0]
            reactions = [instance.idx_to_rid[i] for i in ridx]
        else:
            reactions = []

    elif algorithm.lower() == 'trace':
        X, Y = netExp_trace(R, P, x0, b)  # Assuming this function is already defined
        compounds = instance.create_iteration_dict(X, instance.idx_to_cid)
        reactions = instance.create_iteration_dict(Y, instance.idx_to_rid)
    else:
        raise ValueError('please define algorithm (trace or naive)')

    return compounds, reactions

# define an example helper function for parrellel execution of network expansion with reaction masks
def expansion_helper_reaction_masks(args):
    instance, seedSet, rxns_removed, R, P, b = args
    x0 = instance.initialize_metabolite_vector(seedSet)
    x0 = csr_matrix(x0)
    x0 = x0.transpose()

    yextinct = instance.initialize_reaction_vector(rxns_removed)
    reaction_mask = csr_matrix(np.diag(1 - yextinct))
    Pstar = P * reaction_mask
    Rstar = R * reaction_mask

    x, y = netExp(Rstar, Pstar, x0, b)

    if x.toarray().sum() > 0:
        cidx = np.nonzero(x.toarray().T[0])[0]
        compounds = [instance.idx_to_cid[i] for i in cidx]
    else:
        compounds = []

    if y.toarray().sum() > 0:
        ridx = np.nonzero(y.toarray().T[0])[0]
        reactions = [instance.idx_to_rid[i] for i in ridx]
    else:
        reactions = []

    return compounds, reactions