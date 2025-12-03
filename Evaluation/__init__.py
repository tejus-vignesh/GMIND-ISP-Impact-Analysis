"""
GMIND Evaluation Package.

This package provides comprehensive tools for evaluating object detection and
tracking models on the GMIND dataset. It includes:

- Core evaluation scripts for running detection and tracking
- Analysis tools for detailed performance analysis
- Metric computation utilities
- Visualisation tools for debugging and presentation
- Utility scripts for data preparation and processing

All scripts follow COCO evaluation protocols and use standard formats for
compatibility with other evaluation tools.

Example Usage:
    # Run main evaluation
    from Evaluation.core.baseline_detector_and_tracker import main
    main()

    # Analyse results
    from Evaluation.analysis.analyse_results import main
    main()
"""

__version__ = "1.0.0"
__author__ = "GMIND SDK Team"
