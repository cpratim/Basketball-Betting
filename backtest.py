from data import load_xy
from sklearn.model_selection import train_test_split
from model import MultiRegressor, get_acc, feature_importances
from scipy.optimize import differential_evolution
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def earn(o, amt):
    if o < 0:
        return (100/(abs(o))) * amt * .99
    return ((o)/100) * amt * .99

def create_eq(bor=10, cdr=10, crr=2):

    def f(b, c):
        return (b/bor)
    return f

def backtest(odds, y_pred, y_true, games=2567, balance=1000, disp=False, dif=True, min_confidence=1.5):

    init_balance = balance
    neg = 0
    n = 1
    x, y = [], []
    correct = 1
    for i in range(games):
        o, p, r = odds[i], y_pred[i], y_true[i]
        f = create_eq()
        bet_amt = f(init_balance, p)
        
        balance += e
        if balance < init_balance:
            neg += 1
        n += 1
        x.append(n)
        y.append(balance)
    plt.plot(x, y)
    plt.show()
    if not dif:
        return balance/init_balance*100, correct/n*100, n, balance
    return neg


if __name__ == '__main__':

    X, y, odds = load_xy()
    indexes = list(range(len(odds)))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.3, random_state=5, shuffle=False,
    )

    print(feature_importances(X_train, y_train))

    odds_train, odds_test, ind_train, ind_test = train_test_split(
        odds, indexes, test_size=.3, random_state=5, shuffle=False,
    )

    reg = MultiRegressor(samples=5)
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)

    print(get_acc(y_pred, y_test))
    print(len(y_test), len(odds_test), len(y_pred))
