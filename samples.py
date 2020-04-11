DIGITDATA_DIR = "data/digitdata"

trainingFileName = DIGITDATA_DIR + "/trainingimages"
trainingLabelFileName = DIGITDATA_DIR + "/traininglabels"
testFileName = DIGITDATA_DIR + "/testimages"
testLabelFileName = DIGITDATA_DIR + "/testlabels"

TEST = "TEST"
TRAIN = "TRAIN"

global trainingFileObject
global trainingLabelFileObject
global testFileObject
global testLabelFileObject

global FileObject
global LabelFileObject


def readFiles(typeData):
    if typeData == TRAIN:
        FileObject = open(trainingFileName)
        LabelFileObject = open(trainingLabelFileName)
    elif typeData == TEST:
        FileObject = open(testFileName)
        LabelFileObject = open(testLabelFileName)

    lines_itr = iter(FileObject.readlines())
    labelsLines_itr = iter(LabelFileObject.readlines())

    return lines_itr, labelsLines_itr


def closeFiles():
    FileObject.close()
    LabelFileObject.close()

