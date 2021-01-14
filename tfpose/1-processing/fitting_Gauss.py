from scipy.stats import norm

#fitting Gauss
def funcFitGaus(dfInput):       # dfInput: dataframe 
    mu, std = norm.fit(dfInput)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    return x, p, mu, std

x2, p2, mu2, std1 =md.funcFitGaus( md.funcAge(df2) )
plt.plot(x2, p2, ‘k’, linewidth=2)