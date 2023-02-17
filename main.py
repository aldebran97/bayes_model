from bayes import Article, NaiveBayesTextClassification
from uuid import uuid4


def build_articles():
    result = [Article(uuid4(), '123456af', 'C1'),
              Article(uuid4(), 'asdkahfjahjkfakfhoagj14', 'C2'),
              Article(uuid4(), '234576080705746911111111111111111111ab', 'C1'),
              Article(uuid4(), 'sadkpocsakopckpamlfmng sada23', 'C2'),
              Article(uuid4(), '34850285426240692806aj', 'C1')]
    return result


def try_bernoulli():
    model = NaiveBayesTextClassification(method='Bernoulli')
    model.fit_mul(build_articles())
    model.train()
    print(model.predict(Article(uuid4(), '34850285426240692806A', None)))
    print(model.predict(Article(uuid4(), 'defrtjhrtjA', None)))
    pass


def try_gaussian():
    model = NaiveBayesTextClassification(method='Gaussian')
    model.fit_mul(build_articles())
    model.train()
    print(model.predict(Article(uuid4(), '34850285426240692806A', None)))
    print(model.predict(Article(uuid4(), 'defrtjhrtjA', None)))
    print(model.predict_label(Article(uuid4(), 'defrtjhrtjA', None)))
    pass


if __name__ == '__main__':
    # try_bernoulli()
    try_gaussian()
    pass
