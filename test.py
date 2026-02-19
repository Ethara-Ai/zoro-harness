class CasuelAttention():
    def __init__(embedding_matrix):
        self.linear_embedding = linear_embedding


    def self_attention(sequence):
        ## init?? 
        mask = torch.???()

        x = torch.matmul(sequence, self.linear_embedding)

        x, y, z = x, x, x

        attention = torch.matmul(x, y.transpose(-2, -1))

        if mask:
            attention = torch.mask(attention, mask)

        attention_score = torch.softmax(attention)

        output = attention_score @ z

        return output
    
    
