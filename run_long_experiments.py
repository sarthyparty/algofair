#!/usr/bin/env python3
import sys
import time
import logging
from datetime import datetime
from optimized_analysis import OptimizedStudentAnalysis

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/long_experiment.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def run_long_experiments():
    logger = logging.getLogger(__name__)
    logger.info("Starting long experiments...")
    
    start_time = time.time()
    
    try:
        analysis = OptimizedStudentAnalysis()
        iteration_counts = [10, 25, 50, 100, 200]
        results = analysis.run_experiment_batch(iteration_counts)
        analysis.generate_comparison_plots(results)
        
        total_time = time.time() - start_time
        logger.info(f"Completed in {total_time:.2f} seconds")
        
        for iters in iteration_counts:
            exp = results['gerryfair_experiments'][iters]
            logger.info(f"  {iters:3d} iters: {exp['training_time']:6.2f}s")
        
        logger.info("Run: python process_long_results.py")
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        return False
    
    return True

def run_quick_test():
    logger = logging.getLogger(__name__)
    logger.info("Running quick test...")
    
    try:
        analysis = OptimizedStudentAnalysis()
        results = analysis.run_experiment_batch([3, 7])
        logger.info("Quick test completed successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"Quick test failed: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            success = run_quick_test()
        elif sys.argv[1] == "--long":
            success = run_long_experiments()
        else:
            print("Usage:")
            print("  python run_long_experiments.py --test")
            print("  python run_long_experiments.py --long")
            sys.exit(1)
    else:
        print("Usage:")
        print("  python run_long_experiments.py --test")
        print("  python run_long_experiments.py --long")
        sys.exit(1)
    
    sys.exit(0 if success else 1)