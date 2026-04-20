#!/bin/bash
set -e

case "${1:-train-all}" in

  train-all)
    echo "═══════════════════════════════════════════════"
    echo "  Training ALL models (autoencoder + forecaster)"
    echo "═══════════════════════════════════════════════"
    echo ""
    echo "Step 1/3: Generating synthetic data..."
    python data/generate_synthetic_data.py
    echo ""
    echo "Step 2/3: Training LSTM Autoencoder..."
    python -m model.train
    echo ""
    echo "Step 3/3: Training LSTM Forecaster (synthetic)..."
    python -m model.train_forecaster
    echo ""
    echo "All models trained. Artifacts in saved/"
    ls -la saved/
    ;;

  train-ae)
    echo "Training LSTM Autoencoder..."
    python data/generate_synthetic_data.py
    python -m model.train
    ;;

  train-fc)
    echo "Training LSTM Forecaster (synthetic data)..."
    python -m model.train_forecaster "$@"
    ;;

  train-fc-real)
    echo "Training LSTM Forecaster (real NeonDB data)..."
    shift  # remove 'train-fc-real' from args
    python -m model.train_forecaster_realdata "$@"
    ;;

  check-data)
    echo "Checking real data availability..."
    python -m model.train_forecaster_realdata --check-only
    ;;

  evaluate)
    echo "Running evaluation..."
    python -m model.evaluate
    ;;

  shell)
    exec /bin/bash
    ;;

  *)
    echo "Unknown command: $1"
    echo ""
    echo "Available commands:"
    echo "  train-all      Train everything (default)"
    echo "  train-ae       Train autoencoder only"
    echo "  train-fc       Train forecaster on synthetic data"
    echo "  train-fc-real  Train forecaster on real NeonDB data"
    echo "  check-data     Check if enough real data exists"
    echo "  evaluate       Run model evaluation"
    echo "  shell          Open bash shell"
    exit 1
    ;;

esac