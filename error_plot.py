import matplotlib.pyplot as plt

class Error:

    def __init__(self,dataset,rate,type,method):
        self.rate=rate
        self.dataset=dataset
        self.type=type
        self.method=method
        self.graphplot(self.dataset, self.rate,self.type,self.method)


    def graphplot(self,a,b,c,d):
        p=len(b)
        for i in range(p):
            plt.plot(a,b[i],label=c[i])
            plt.xlim(500,5000)
            plt.ylim(0,100)
        plt.title(d)
        plt.legend()
        plt.show()