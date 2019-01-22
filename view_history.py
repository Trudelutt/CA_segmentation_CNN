import json
import matplotlib.pyplot as plt

def load_history(path):
    with open(path) as json_data:
        d = json.load(json_data)
        return d

def plot_history(history, name, save=False):
    plt.figure()
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title(name+' loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    if save:
        plt.savefig('fig/'+name+'_loss.png')
        plt.close()
    plt.figure()
    metric_keys = ['recall', 'binary_accuracy', 'dsc', 'precision']
    for key in metric_keys:
        plt.plot(history[key])
    plt.title(name+' metrics')
    plt.ylabel('value')
    plt.xlabel('epoch')
    plt.legend(metric_keys, loc='upper left')
    if save:
        plt.savefig('fig/'+name+'_metric.png')
        plt.close()

    plt.figure()
    val_metric_keys = ['val_recall', 'val_binary_accuracy', 'val_dsc', 'val_precision']
    for key in val_metric_keys:
        plt.plot(history[key])
    plt.title(name+' val metrics')
    plt.ylabel('value')
    plt.xlabel('epoch')
    plt.legend(val_metric_keys, loc='upper left')
    if not save:
        plt.show()
    else:
        plt.savefig('fig/'+name+'_val_metric.png')
        plt.close()

if __name__=="__main__":
    d = load_history("./history/BVNet_LM.json")
    print(d.keys())
    plot_history(d, "BVNet LM", save= True)
