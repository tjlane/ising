BJs = np.linspace(0.1, 0.5, 21)
sz = (64,64)
steps = 10000000

for BJ in [BJs.min(), BJs.max()]:
    print(BJ)
    m = Model(sz, BJ, save_interval=1000, verbose=False)
    m.mc_steps(steps)
    m.file.close()
