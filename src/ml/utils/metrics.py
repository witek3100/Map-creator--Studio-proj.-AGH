

def accuracy(pred, targ):
    pos = 0
    sum = 0
    for pred, targ in zip(pred, targ):
        if float(round(pred.item())) == targ.item():
            pos += 1
        sum += 1
    return pos/sum