from Evaluators.prequential_simulation_evaluator import PrequentialSimulationEvaluator

# To run other evaluators, change this class
evaluator_class = PrequentialSimulationEvaluator

# Create evaluator and run it
evaluator = evaluator_class()
evaluator.evaluate()
