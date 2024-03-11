from rdkit import Chem
from rdkit.Chem import Descriptors

def smiles_to_physical_properties(smiles):
    """
    Convert a SMILES string into a vector of physical-chemical properties.

    Args:
    - smiles (str): The SMILES string.

    Returns:
    - properties (numpy array): The physical-chemical properties.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    properties = [
        Descriptors.MolWt(mol),                # Molecular weight
        Descriptors.ExactMolWt(mol),           # Exact molecular weight
        Descriptors.MolLogP(mol),              # LogP (octanol/water partition coefficient)
        Descriptors.MolMR(mol),                # Molecular refractivity
        Descriptors.NumValenceElectrons(mol),  # Number of valence electrons
        Descriptors.TPSA(mol),                 # Topological polar surface area
        Descriptors.NOCount(mol),              # Number of nitrogen and oxygen atoms
        Descriptors.NumHDonors(mol),           # Number of hydrogen bond donors
        Descriptors.NumHAcceptors(mol),        # Number of hydrogen bond acceptors
        Descriptors.NumRotatableBonds(mol),    # Number of rotatable bonds
        Descriptors.NumAromaticRings(mol),     # Number of aromatic rings
        Descriptors.NumAliphaticRings(mol),    # Number of aliphatic rings
        Descriptors.NumSaturatedRings(mol),    # Number of saturated rings
        Descriptors.FractionCSP3(mol),         # Fraction of sp3-hybridized carbons
        Descriptors.NumAromaticCarbocycles(mol),  # Number of aromatic carbocycles
        Descriptors.NumSaturatedCarbocycles(mol), # Number of saturated carbocycles
        Descriptors.NumAliphaticCarbocycles(mol)  # Number of aliphatic carbocycles
    ]
    return properties

# Define the SMILES string
v_d = "OC1C(N2C=3C(N=C2)=C(N)N=CN3)OC(C[S+](CCC(C([O-])=O)N)C)C1O"

# Convert the SMILES string to physical-chemical properties
properties = smiles_to_physical_properties(v_d)

print("Physical-chemical properties:", properties)
