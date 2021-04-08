from keras_flops import get_flops
import model.models as mymodels

# Calculae FLOPS
def getflops(model):
    flops = get_flops(model, batch_size=1)
    g = flops / 2.0 / 10 ** 9
    # print(f"FLOPS: {flops / 10 ** 9:.03} G")
    print('{:.3f}G'.format(g))
    return g

if __name__ == '__main__':
    hwd = 224
    class_total = 1000
    model = mymodels.myEfficientNetB0(input_shape=(hwd, hwd, 3), classes=class_total)
    flops = getflops(model)
    print(flops)

# 2021-04-09 guangjinzheng
