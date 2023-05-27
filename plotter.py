from pandas import read_csv
import matplotlib.pyplot as plt

if __name__ == "__main__":
    df = read_csv('dynaq.csv')
    dynq = df.plot(x='episode', y='steps', label='dynaq')
    dynq.set_xlabel("episodes")
    dynq.set_ylabel("steps")
    dynq.set_title(f"DynaQ\nAlpha: 1.0 Gamma: 0.95 Épsilon: 0.1")
    plt.show()

    success = read_csv('success_rate.csv')
    suc = success.plot.scatter(x='episode', y='success', label='success')
    suc.set_xlabel("episodes")
    suc.set_ylabel("steps")
    plt.show()
    
    
