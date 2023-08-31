import matplotlib.pyplot as plt
import re

def read_log(log_path):
    with open(log_path, 'r', encoding='utf-8') as f:
        log = f.readlines()
    return log

def get_loss(log):
    loss = []
    dev_loss = []
    dev_bleu = []
    epoch_loss_pattern = re.compile(r"Epoch: (\d+), loss: ([\d.]+)")
    epoch_dev_pattern = re.compile(r"Epoch: (\d+), Dev loss: ([\d.]+), Bleu Score: ([\d.]+)")

    for line in log:
        matches = epoch_loss_pattern.findall(line)
        for match in matches:
            epoch, loss_ = match
            loss.append(float(loss_))
        matches = epoch_dev_pattern.findall(line)
        for match in matches:
            epoch, dev_loss_, bleu_ = match
            dev_loss.append(float(dev_loss_))
            dev_bleu.append(float(bleu_))
    return loss, dev_loss, dev_bleu

if __name__ == '__main__':
    log_path = './experiment/train.log'
    # log_path = './experiment/train_lora_ted.log'
    log = read_log(log_path)
    loss, dev_loss, dev_bleu = get_loss(log)
    
    # plot loss x = epoch:int, y = loss
    x = list(range(1, len(loss)+1))
    x = [int(i) for i in x]
    plt.plot(x, loss)
    plt.plot(x, dev_loss)
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Dev'], loc='upper left')
    plt.show()

    # plot bleu x = epoch, y = bleu
    plt.plot(dev_bleu)
    plt.title('Corpus Bleu')
    plt.ylabel('Corpus Bleu')
    plt.xlabel('Epoch')
    plt.legend(['Dev'], loc='upper left')
    plt.show()
    