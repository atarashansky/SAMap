"""Integration tests for SAMAP using example data."""

from pathlib import Path

import pytest
from samalg import SAM

from samap import SAMAP
from samap.analysis import GenePairFinder, get_mapping_scores, sankey_plot

# Path to example data relative to repo root
EXAMPLE_DATA = Path(__file__).parent.parent.parent / "example_data"


@pytest.fixture(scope="module")
def example_data_path():
    """Return path to example data, skip if not available."""
    if not EXAMPLE_DATA.exists():
        pytest.skip("Example data not available")
    return EXAMPLE_DATA


@pytest.fixture(scope="module")
def sam_objects(example_data_path):
    """Load SAM objects from example h5ad files."""
    sam1 = SAM()
    sam1.load_data(str(example_data_path / "planarian.h5ad"))

    sam2 = SAM()
    sam2.load_data(str(example_data_path / "schistosome.h5ad"))

    sam3 = SAM()
    sam3.load_data(str(example_data_path / "hydra.h5ad"))

    return {"pl": sam1, "sc": sam2, "hy": sam3}


class TestSAMAPInitialization:
    """Test SAMAP initialization with example data."""

    def test_samap_init_with_three_species(self, sam_objects, example_data_path):
        """Test SAMAP initialization with three species (pl, sc, hy)."""
        sm = SAMAP(
            sam_objects,
            f_maps=str(example_data_path / "maps") + "/",
        )

        assert sm is not None
        assert hasattr(sm, "sams")
        assert len(sm.sams) == 3

    def test_samap_init_with_two_species(self, sam_objects, example_data_path):
        """Test SAMAP initialization with two species subset."""
        two_species = {"pl": sam_objects["pl"], "sc": sam_objects["sc"]}

        sm = SAMAP(
            two_species,
            f_maps=str(example_data_path / "maps") + "/",
        )

        assert sm is not None
        assert len(sm.sams) == 2


class TestSAMAPMapping:
    """Test SAMAP mapping functionality."""

    @pytest.fixture(scope="class")
    def samap_instance(self, sam_objects, example_data_path):
        """Create a SAMAP instance for mapping tests."""
        return SAMAP(
            sam_objects,
            f_maps=str(example_data_path / "maps") + "/",
        )

    def test_run_mapping(self, samap_instance):
        """Test running the SAMAP algorithm."""
        samap_instance.run()

        # Verify mapping completed
        assert hasattr(samap_instance, "samap")

    def test_get_mapping_scores(self, samap_instance):
        """Test retrieving mapping scores after running SAMAP."""
        # Ensure mapping has been run
        if not hasattr(samap_instance, "samap"):
            samap_instance.run()

        # Test getting mapping scores using annotation columns
        # These column names come from the example data
        keys = {"pl": "cluster", "hy": "Cluster", "sc": "tissue"}
        D, MappingTable = get_mapping_scores(samap_instance, keys, n_top=0)

        assert D is not None
        assert MappingTable is not None
        # D and MappingTable should be DataFrames
        assert hasattr(D, "index")
        assert hasattr(MappingTable, "index")
        assert hasattr(MappingTable, "columns")

    def test_sankey_plot(self, samap_instance):
        """Test generating a sankey plot from mapping scores."""
        # Ensure mapping has been run
        if not hasattr(samap_instance, "samap"):
            samap_instance.run()

        # Get mapping scores
        keys = {"pl": "cluster", "hy": "Cluster", "sc": "tissue"}
        D, MappingTable = get_mapping_scores(samap_instance, keys, n_top=0)

        # Generate sankey plot
        fig = sankey_plot(MappingTable, species_order=["pl", "sc", "hy"], align_thr=0.1)

        # Verify it's a plotly figure
        import plotly.graph_objects as go

        assert isinstance(fig, go.Figure)

        # Verify the sankey has nodes and links
        assert len(fig.data) == 1
        assert fig.data[0].node.label is not None
        assert len(fig.data[0].node.label) > 0
        assert fig.data[0].link.source is not None

    def test_gene_pair_finder(self, samap_instance):
        """Test GenePairFinder to find genes linking cell types."""
        # Ensure mapping has been run
        if not hasattr(samap_instance, "samap"):
            samap_instance.run()

        # Create GenePairFinder with all 3 species
        keys = {"pl": "cluster", "sc": "tissue", "hy": "Cluster"}
        gpf = GenePairFinder(samap_instance, keys=keys)

        # Find genes linking two cell types
        n1 = "pl_Neoblast: 0"
        n2 = "sc_Neoblast"
        Gp, G1, G2, pvals1, pvals2 = gpf.find_genes(n1, n2)

        # Verify results
        assert Gp is not None
        assert G1 is not None
        assert G2 is not None
        # Gp contains gene pairs, G1/G2 are individual gene lists
        assert len(Gp) > 0
        assert len(G1) > 0
        assert len(G2) > 0
        assert len(pvals1) == len(Gp)
        assert len(pvals2) == len(Gp)
