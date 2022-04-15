import datacollection as datacollection
import model as model
import evaluation as evaluation


def main():
    print("Welcome to LSF recognition")
    print("Preparating global variables...")
    datacollection.init_video_variables()
    print("Preparating folders...")
    datacollection.folder_preparation()

    # Record data
    print("Starting to analyse data...")
    # datacollection.record_data()
    datacollection.analyse_data()

    print("Building and training the model...")
    model.start_model()
    print("Live predcition")
    evaluation.realtime_prediction()


if __name__ == "__main__":
    main()
