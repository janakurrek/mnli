class Parameters():
    def __init__(self):
        # gpu
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # word vectors
        self.embed_size = 50
        self.word_vectors = True
        self.glove_path = '/home/ndg/users/jkurre/mnli/utils/embeddings/glove.6B.50d.txt'
        # model configs
        self.hidden_size = 1024
        self.batch_size = 32
        self.input_size = 76790
        self.output_size = 4
        self.n_layers = 2
        self.n_cells = 4
        self.dropout = 0.5
        # training
        self.epochs = 40
        self.learning_rate = 0.0001
        self.outpath = '/home/ndg/users/jkurre/mnli/models/bilstm_revised_attention_40epochs.pt'
        self.outfile = '/home/ndg/users/jkurre/mnli/models/outputs/bilstm_with_attention_40epochs.txt'
        # export settings
        self.load = False
        self.save_model = False
        self.loadpath = '/home/ndg/users/jkurre/mnli/models/bilstm_revised_attention_40epochs.pt'