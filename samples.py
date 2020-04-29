class Samples:

    def __init__(self, DATA_DIR):
        self.TestFileObject = None
        self.TestLabelFileObject = None
        self.TrainFileObject = None
        self.TrainLabelFileObject = None
        self.ValidationFileObject = None
        self.ValidationLabelFileObject = None

        self.test_lines_itr = None
        self.test_labelsLines_itr = None
        self.train_lines_itr = None
        self.train_labelsLines_itr = None
        self.validate_lines_itr = None
        self.validate_labelsLines_itr = None


        self.trainingFileName = DATA_DIR + "/finalimages"
        self.trainingLabelFileName = DATA_DIR + "/finallabels"
        self.testFileName = DATA_DIR + "/testimages"
        self.testLabelFileName = DATA_DIR + "/testlabels"
        self.validationFileName = DATA_DIR + "/validationimages"
        self.validationLabelFileName = DATA_DIR + "/validationlabels"

        TEST = "TEST"
        TRAIN = "TRAIN"
        VALIDATION = "VALIDATION"

    # def open_many_files(self,file_name):
    #     with open(self.name, 'r') as f:
    #         c = 0
    #         for l in f:
    #             c+=1
    #         l=c+1-self.k_value
    #         for i in range(0,l):
    #             lines = [line for line in f][:self.k_value]
    #             object=lines
    #         return object


    def closeFiles(self):
        self.TestFileObject.close()
        self.TestLabelFileObject.close()
        self.TrainFileObject.close()
        self.TrainLabelFileObject.close()
        self.ValidationFileObject.close()
        self.ValidationLabelFileObject.close()

    def initTestIters(self):
        self.TestFileObject.close()
        self.TestLabelFileObject.close()
        self.TestFileObject = open(self.testFileName)
        self.TestLabelFileObject = open(self.testLabelFileName)
        self.test_lines_itr = iter(self.TestFileObject.readlines())
        self.test_labelsLines_itr = iter(self.TestLabelFileObject.readlines())

    def initValidateIters(self):
        self.validate_lines_itr = iter(self.ValidationFileObject.readlines())
        self.validate_labelsLines_itr = iter(self.ValidationLabelFileObject.readlines())

    def readFiles(self):
        self.TrainFileObject = open(self.trainingFileName)
        self.TrainLabelFileObject = open(self.trainingLabelFileName)
        self.TestFileObject = open(self.testFileName)
        self.TestLabelFileObject = open(self.testLabelFileName)
        self.ValidationFileObject = open(self.validationFileName)
        self.ValidationLabelFileObject = open(self.validationLabelFileName)

        self.train_lines_itr = iter(self.TrainFileObject.readlines())
        self.train_labelsLines_itr = iter(self.TrainLabelFileObject.readlines())

        self.test_lines_itr = iter(self.TestFileObject.readlines())
        self.test_labelsLines_itr = iter(self.TestLabelFileObject.readlines())

        self.initValidateIters()