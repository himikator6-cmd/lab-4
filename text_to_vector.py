import gensim




def W2V_Skip_gram(Documents, Vector_Size, Window):
    model = gensim.models.Word2Vec(sentences = Documents, vector_size = Vector_Size, window = Window,
                                   min_count = 5, epochs = 10, sg = 1)
    model.init_sims(replace = True)
    return model


def W2V_CBOW(Documents, Vector_Size, Window):
    model = gensim.models.Word2Vec(sentences = Documents, vector_size = Vector_Size, window = Window,
                                   min_count = 5, epochs = 10, sg = 0)
    model.init_sims(replace = True)
    return model


def Fasttext_Skip_gram(Documents, Vector_Size, Window):
    model = gensim.models.FastText(sentences = Documents, vector_size = Vector_Size, window = Window,
                                   min_count = 5, epochs = 10, sg = 1)
    model.init_sims(replace = True)
    return model


def Fasttext_CBOW(Documents, Vector_Size, Window):
    model = gensim.models.FastText(sentences = Documents, vector_size = Vector_Size, window = Window,
                                   min_count = 5, epochs = 10, sg = 0)
    model.init_sims(replace = True)
    return model
