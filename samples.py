DIGITDATA_DIR = "data/digitdata"
FACEDATA_DIR = "data/facedata"

trainingFileName = DIGITDATA_DIR + "/trainingimages"
trainingLabelFileName = DIGITDATA_DIR + "/traininglabels"
testFileName = DIGITDATA_DIR + "/testimages"
testLabelFileName = DIGITDATA_DIR + "/testlabels"
validationFileName = DIGITDATA_DIR + "/validationimages"
validationLabelFileName = DIGITDATA_DIR + "/validationlabels"

TEST = "TEST"
TRAIN = "TRAIN"
VALIDATION = "VALIDATION"


class Samples:
    def __init__(self):
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
        self.TestFileObject = open(testFileName)
        self.TestLabelFileObject = open(testLabelFileName)
        self.test_lines_itr = iter(self.TestFileObject.readlines())
        self.test_labelsLines_itr = iter(self.TestLabelFileObject.readlines())

    def initValidateIters(self):
        self.validate_lines_itr = iter(self.ValidationFileObject.readlines())
        self.validate_labelsLines_itr = iter(self.ValidationLabelFileObject.readlines())

    def readFiles(self):
        self.TrainFileObject = open(trainingFileName)
        self.TrainLabelFileObject = open(trainingLabelFileName)
        self.TestFileObject = open(testFileName)
        self.TestLabelFileObject = open(testLabelFileName)
        self.ValidationFileObject = open(validationFileName)
        self.ValidationLabelFileObject = open(validationLabelFileName)

        self.train_lines_itr = iter(self.TrainFileObject.readlines())
        self.train_labelsLines_itr = iter(self.TrainLabelFileObject.readlines())

        self.test_lines_itr = iter(self.TestFileObject.readlines())
        self.test_labelsLines_itr = iter(self.TestLabelFileObject.readlines())

        self.initValidateIters()