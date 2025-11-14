"""SDA (Synthetic Data Assessment) client for data quality report generation.

This client wraps the data_quality_report package to generate comparative
assessment reports between real and synthetic datasets.
"""

import logging
from typing import Any, Dict, Optional
import rockfish as rf
from data_quality_report.check import Assessor
from rockfish.dataset import LocalDataset
import tempfile
from pathlib import Path
# Set matplotlib to non-interactive backend BEFORE importing pyplot
# This prevents GUI windows and Python icon from appearing in dock
import matplotlib
matplotlib.use('Agg')

logger = logging.getLogger(__name__)


class SDAClient:
    """Client for generating data quality assessment reports using SDA.

    This client uses the Assessor class from data_quality_report to compare
    real and synthetic datasets and generate comprehensive HTML reports.
    """

    # Default values 
    DEFAULT_DATASET_ID = "2cLvK53WVcpNkRtLqM8dkn"
    DEFAULT_SYN_ID = "4rIpjTeMxTeaLCxV9tQ1if"
    DEFAULT_OUTPUT_FILE = "test1.html"

    # Default configuration using RTF (always used, user input ignored for now)
    DEFAULT_CONFIG = {
        "encoder": {
            "metadata": [
                {"field": "released_year", "type": "categorical"},
                {"field": "released_month", "type": "categorical"},
                {"field": "released_day", "type": "categorical"},
                {"field": "key", "type": "categorical"},
                {"field": "mode", "type": "categorical"},
                {"field": "in_spotify_playlists", "type": "continuous"},
                {"field": "bpm", "type": "continuous"}
            ]
        },
        "rtf": {
            "mode": "tabular",
            "num_bootstrap": 2,
            "tabular": {
                "epochs": 1,
                "transformer": {
                    "gpt2_config": {
                        "layer": 1,
                        "head": 1,
                        "embed": 1
                    }
                }
            }
        }
    }


    def __init__(self):
        """Initialize SDA client with a Rockfish connection.

        """
        # TODO: this is temp and hardcoded for demo
        self._conn = rf.Connection.from_config(profile = "prod", project="7H70G3UctpINoletOBbVCs")
        logger.info("SDA client initialized")

    # TODOs: currently not used, always default
    def _auto_generate_config(self, dataset: LocalDataset) -> Dict[str, Any]:
        """Auto-generate config from dataset by detecting categorical fields.

        Args:
            dataset: Local dataset to analyze

        Returns:
            Configuration dict with encoder metadata and tabular-gan settings
        """
        # Convert to pandas to detect types
        df = dataset.to_pandas()

        # Detect categorical fields (object dtype)
        categorical_fields = df.select_dtypes(include=["object"]).columns.tolist()

        # All other fields are continuous
        all_fields = dataset.table.column_names
        continuous_fields = [f for f in all_fields if f not in categorical_fields]

        # Build metadata list
        metadata = []
        for field in categorical_fields:
            metadata.append({"field": field, "type": "categorical"})
        for field in continuous_fields:
            metadata.append({"field": field, "type": "continuous"})

        # Get record count from dataset
        record_count = len(df)

        config = {
            "encoder": {
                "metadata": metadata
            },
            "tabular-gan": {
                "epochs": 100,
                "records": record_count
            }
        }

        logger.info(f"Auto-generated config with {len(categorical_fields)} categorical "
                   f"and {len(continuous_fields)} continuous fields, {record_count} records")
        return config

    async def generate_report(
        self,
        dataset_id: Optional[str] = None,
        syn_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate a data quality assessment report comparing real and synthetic datasets.

        Args:
            dataset_id: ID of the real dataset (default: uses DEFAULT_DATASET_ID)
            syn_id: ID of the synthetic dataset (default: uses DEFAULT_SYN_ID)
            config: Configuration dict (TEMP - DEFAULT_CONFIG is always used)
            output_file: Output HTML filename (default: "test1.html", saved to ~/Documents/rockfish-reports/)

        Returns:
            Dict with report generation results including:
                - success: Whether report was generated successfully
                - report_path: Path to the generated HTML file
                - datasets_compared: IDs of datasets that were compared
                - config_used: Configuration that was used
        """
        # Use defaults if not provided
        dataset_id = self.DEFAULT_DATASET_ID
        syn_id = self.DEFAULT_SYN_ID
        output_file = self.DEFAULT_OUTPUT_FILE

        # Create report directory in user's home directory or temp, not in current dir
        # Use ~/rockfish-reports for better accessibility
        report_dir = Path.home() / "Documents" / "rockfish-reports"
        try:
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir / output_file
        except Exception as e:
            # Fallback to temp directory if home directory fails
            logger.warning(f"Could not create report dir in home: {e}, using temp directory")
            report_dir = Path(tempfile.gettempdir()) / "rockfish-reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir / output_file

        logger.info(f"Generating quality report: real={dataset_id}, syn={syn_id}, output={report_path}")

        try:
            # Load real dataset
            logger.info(f"Loading real dataset: {dataset_id}")
            dataset = await rf.Dataset.from_id(self._conn, dataset_id)
            dataset = await dataset.to_local(self._conn)
            logger.info(f"Real dataset loaded: {len(dataset.to_pandas())} rows")

            # Load synthetic dataset
            logger.info(f"Loading synthetic dataset: {syn_id}")
            syn = await rf.Dataset.from_id(self._conn, syn_id)
            syn = await syn.to_local(self._conn)
            logger.info(f"Synthetic dataset loaded: {len(syn.to_pandas())} rows")

            # Always use DEFAULT_CONFIG (user input ignored for now)
            config = self.DEFAULT_CONFIG
            logger.info("Using DEFAULT_CONFIG with RTF configuration")

            # Create Assessor instance
            logger.info("Creating Assessor instance")
            sda = Assessor(dataset, syn, config=config)

            # Generate report - pass the full path as a string
            logger.info(f"Generating report to: {report_path}")
            await sda.report(output_file=str(report_path), conn=self._conn)

            logger.info(f"Report generated successfully: {report_path}")

            return {
                "success": True,
                "report_path": str(report_path),
                "datasets_compared": {
                    "real": dataset_id,
                    "synthetic": syn_id
                },
                "config_used": config,
                "message": f"Quality assessment report generated successfully at {report_path}"
            }

        except Exception as e:
            logger.error(f"Error generating report: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to generate report: {str(e)}"
            }
