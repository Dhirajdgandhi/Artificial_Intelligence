import perceptron
import samples

if __name__ == '__main__':
    imageLineItr, labelLineItr = samples.readFiles("TRAIN")

    scanInput(imageLineItr, labelLineItr, isTrain=True)

    imageLineItr, labelLineItr = samples.readFiles("TEST")

    error, total = scanInput(imageLineItr, labelLineItr, isTrain=False)
    print("Error is", error, "out of Total of ",total)
    print(error * 100 / total, "%")

    samples.closeFiles()