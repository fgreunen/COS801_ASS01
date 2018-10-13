from Shared import MNIST, FMNIST, CIFAR10, CIFAR100, Utils
import matplotlib.pyplot as plt
import pandas as pd
from decimal import Decimal

def getDataFrame(url):
    runs = Utils.ResultsManager().load(url)
    raw = [(run.trainingAccuracies, run.learningRate, run.elapsed, run.step, run.validationAccuracies) for run in runs]
    maxLength = max([len(x[0]) for x in raw])
    trainingAccuracies = pd.DataFrame(index=range(maxLength))
    validationAccuracies = pd.DataFrame(index=range(maxLength))
    for i in range(len(raw)):
        toAdd = maxLength - len(raw[i][0])
        ta = raw[i][0]
        va = raw[i][4]
        if toAdd > 0:
            ta.extend([None]*toAdd)
            va.extend([None]*toAdd)
        trainingAccuracies['{0:.2f}'.format(raw[i][1])] = pd.Series(ta)
        validationAccuracies['{0:.2f}'.format(raw[i][1])] = pd.Series(va)
        
    runs = Utils.ResultsManager().load(url)
    steps = [x.step for x in runs]
    elapsed = ['{0:.2f}s'.format(x.elapsed) for x in runs]
    learningRate = ['{0:.2f}'.format(x.learningRate) for x in runs]
    trainingAccuracy = ['{0:.4f}'.format(x.trainingAccuracy) for x in runs]
    validationAccuracy = ['{0:.4f}'.format(x.validationAccuracies[-1:][0]) for x in runs]
    testAccuracy = ['{0:.4f}'.format(x.testAccuracy) for x in runs] 
    
    return pd.DataFrame({
            'Steps':steps,
            'Learning Rate':learningRate,
            'Elapsed (seconds)':elapsed,
            'Training Accuracy (%)':trainingAccuracy,
            'Validation Accuracy (%)':validationAccuracy,
            'Test Accuracy (%)':testAccuracy,
    }), trainingAccuracies, validationAccuracies

    
def display(ta, va, testAccuracy, column, textleft=30, textTop=80,xlims=None,ylims=None,legendPos='bottom right'):
    ta = ta[column].dropna()*100
    va = va[column].dropna()*100
    x = range(len(ta))
    plt.plot(x, ta, marker='.', color='#770000', label='Training', linewidth=0.6)
    plt.plot(x, va, marker='', color='#444444', label='Validation', linewidth=2)
    plt.plot(len(ta),testAccuracy*100,'X',color='#008811',markersize=14) 
    if xlims != None:
        plt.xlim(xlims[0],xlims[1])
    if ylims != None:
        plt.ylim(ylims[0],ylims[1])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend(loc=legendPos)
    plt.text(textleft,textTop,'Test Accuracy: {0:.2f}%'.format(testAccuracy*100), horizontalalignment='left', size='small', color='#222222')
    plt.show()

print('MODEL 1')        
df, ta, va = getDataFrame('m1.mnist.json')
display(ta,va,Decimal(df['Test Accuracy (%)'].max()),'0.27',30,80)
df, ta, va = getDataFrame('m1.fmnist.json')
display(ta,va,Decimal(df['Test Accuracy (%)'].max()),'0.27',30,70)
df, ta, va = getDataFrame('m1.cifar10.json')
display(ta,va,Decimal(df['Test Accuracy (%)'].max()),'0.42',0,45)
df, ta, va = getDataFrame('m1.cifar100.json')
display(ta,va,Decimal(df['Test Accuracy (%)'].max()),'0.42',18,25)

print('MODEL 2')     
df, ta, va = getDataFrame('m2.mnist.relu.json')
display(ta,va,Decimal(df['Test Accuracy (%)'].max()),'0.42',125,99.25,(0,225),(95,100.15))
df, ta, va = getDataFrame('m2.mnist.sigmoid.json')
display(ta,va,Decimal(df['Test Accuracy (%)'].max()),'0.18',275,98.6,(200,405),(77,100))

df, ta, va = getDataFrame('m2.fmnist.relu.json')
display(ta,va,Decimal(df['Test Accuracy (%)'].max()),'0.27',100,82,(0,185),(70,100.15))
df, ta, va = getDataFrame('m2.fmnist.sigmoid.json')
display(ta,va,Decimal(df['Test Accuracy (%)'].max()),'0.42',180,36,(75,231),(35,72))

df, ta, va = getDataFrame('m2.cifar10.relu.json')
display(ta,va,Decimal(df['Test Accuracy (%)'].max()),'0.18',95,37,(0,223),(30,95))
df, ta, va = getDataFrame('m2.cifar10.sigmoid.json')
display(ta,va,Decimal(df['Test Accuracy (%)'].max()),'0.65',150,11,(0,255),(5,58))

df, ta, va = getDataFrame('m2.cifar100.relu.json')
display(ta,va,Decimal(df['Test Accuracy (%)'].max()),'0.18',110,8,(0,220),(0,60))
df, ta, va = getDataFrame('m2.cifar100.sigmoid.json')
display(ta,va,Decimal(df['Test Accuracy (%)'].max()),'0.65',260,0.5,(0,410),(-0.25,15))

print('MODEL 3')  
df, ta, va = getDataFrame('m4.mnist.relu.json')
display(ta,va,Decimal(df['Test Accuracy (%)'].max()),'0.20',50,85,(0,125),(70,100.25))
df, ta, va = getDataFrame('m4.fmnist.relu.json')
display(ta,va,Decimal(df['Test Accuracy (%)'].max()),'0.20',40,78,(0,98),(70,100.25))
df, ta, va = getDataFrame('m4.cifar10.relu.json')
display(ta,va,Decimal(df['Test Accuracy (%)'].max()),'0.20',40,30,(0,99),(20,100.25))
df, ta, va = getDataFrame('m4.cifar100.relu.json')
display(ta,va,Decimal(df['Test Accuracy (%)'].max()),'0.20',40,10,(0,83),(5,65))


#latexTable = df.to_latex() 
#df['Elapsed (seconds)']
#df['Validation Accuracy (%)']
#df['Test Accuracy (%)']
#df['Learning Rate']
#display(ta,va,'0.27')


#plt.style.use('seaborn-darkgrid')
#my_dpi=96
#plt.figure(figsize=(480/my_dpi, 480/my_dpi), dpi=my_dpi)
# 
#for column in df:
#   plt.plot(df.index, df[column], marker='', color='grey', linewidth=1, alpha=0.4)
#
#num=0
#for i in df.values[30][1:]:
#   num+=1
#   name=list(df)[num]
#   plt.text(10.2, i, name, horizontalalignment='left', size='small', color='grey')
# 
#plt.ylim(0.8, 1)
## Add titles
#plt.title("Fooking hel", loc='left', fontsize=12, fontweight=0, color='orange')
#plt.xlabel("Epochs")
#plt.ylabel("Test Accuracy")





