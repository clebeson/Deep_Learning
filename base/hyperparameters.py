
class Hyperparameters:
        def __init__(self):
            
            #Default parameters for  dataset
            self.height = 32 #The height of the image (The distance from the top to the bottom)
            self.width = 32   #The width of the image (The distance between the two sides of the image)
            self.channels = 3 #Number of channels in the image. Ex.: for a RGB image this number must be "3", for a grayscaled, "1".
            self.auto_crop = (0, 0) # It signs whether the data augmentation will proceed with auto cropping or not.
            self.data_augmentation = False #It signs whether the data augmentation procedure must be applied or not. 
            self.num_classes = 10 #Number of classes of the data
            self.class_names = [] #The name of aech label 
            self.bottleneck = True #If it is "true" and the the "fine_tunning" is false, the bottleneck procedure wll be applied
            
            #default parameters for the optimizer
            self.initial_learning_rate = 1e-4 #The initial learning rate. This value will decay considering the next two parameters 
            self.decay_steps = 1e3 #Number of steps in which the learning rate will decay
            self.decay_rate = 0.98 #Rate which the leaning rate must decay
            self.batch_size = 128 #Size of mini batch for traning
            self.num_epochs = 200 #Amount of epoch of the training
            self.checkpoints_path= "./tensorboard" #When the model will be  saved
            self.fine_tunning = False  #If it is "true" the gradient will be stopped at the CNN cut layer.
            self.regularizer_type = None  #Could be "1" (L1) or "2" (L2). If "None" or "0" the regularizer will be disabled.
            self.regularizer_scale = 0.005 #Represents the regularizer scale
            
            
            #default parameters for the model
            self.cut_layer = "pool5" #When the CNN will be cut
            self.hidden_layers = [512] # Also used for creating autoencoders layers. If it is empty ([]) there will not be 
                                       #fully connected layers in the model, just the convolutional ones (CNN).
            self.normalization = None  #can be "cos_norm" or "batch_norm". If it is "None" the normalization will be disabled. 
            self.keep = None  #dropout keep probability. If it is "None", "0" or "1" the dropout will be disabled.
            self.activation = "relu" #Activation of the fully connected layers. could be "sigmoid" or "None". If it is "None" 
                                     #the activation will be disabled.
            self.batch_norm = None #Set a batch normalization
            
            


        
        
        

        