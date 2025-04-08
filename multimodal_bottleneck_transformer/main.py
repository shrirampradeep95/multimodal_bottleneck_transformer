# Import relevant libraries
import datetime
from utils.global_vars import parameters

# Import useer defined library
from utils.mbt_runner_class import MBTRunnerClass


if __name__ == '__main__':
    print("Starting run for MBT module at {}".format(datetime.datetime.now()))

    # Initialise the parameters for the model
    parameters = parameters.copy()
    parameters["dataset"] = "AudioSet"
    parameters["videos_to_use"] = 20
    parameters["modality"] = 'av'
    parameters["epochs"] = 2

    # Initialise MBT runner model class
    runner = MBTRunnerClass(parameters)
    runner.load_format_data()
