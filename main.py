import datacollection as datacollection
import model as model
import configuration
import evaluation as evaluation


def main():
    #model.getMaxNumberFrame()
    #print("Max number of frame = " + str(configuration.maxNumberFrame))

    print("Welcome to LSF recognition")
    print("Preparating global variables...")
    datacollection.init_video_variables()
    print("DEBUG action_wanted length : "+str(len(configuration.actions_wanted)))
    print("Preparating folders...")
    #datacollection.folder_preparation()
    # Debug
    print(str(configuration.actions))

    # Record data
    print("Starting to analyse data...")
    # datacollection.record_data()
    #datacollection.analyse_data()

    print("Building and training the model...")
    model.start_model()
    print("Live predcition")
    evaluation.realtime_prediction()

if __name__ == "__main__":
    main()
