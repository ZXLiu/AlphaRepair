import random
import torch


# 基本波束搜索类(寻找概率最高的前N个补丁)
class BeamSearch(object):

    def __init__(self, model, tokenizer, masked_code, device, beam_width=10, re_rank=False):
        # parameters
        self.model = model
        self.tokenizer = tokenizer
        self.masked_code = masked_code
        self.beam_width = beam_width
        self.device = device
        self.re_rank = re_rank
        # calculation
        self.inputs = self.tokenizer(self.masked_code, return_tensors='pt').to(self.device)
        self.input_ids = self.inputs["input_ids"]
        self.attention_mask = self.inputs['attention_mask']
        # 用于查找标记化后输入序列中屏蔽标记的索引
        self.masked_index = torch.flatten(
            torch.nonzero(self.input_ids[0] == self.tokenizer.mask_token_id, as_tuple=False))

    def __beam_generate(self, prev_tokens, current_masked_index):  # 生成current_masked_index这个掩码位置的概率最高的前beam_width个补丁
        new_input = self.input_ids.clone().detach()  # 创建副本，并且将其从计算图中分离出来
        for i, token in enumerate(prev_tokens):
            if token is not None:
                new_input[0][self.masked_index[i]] = token

        outputs = self.model(input_ids=new_input, attention_mask=self.attention_mask)
        probs = outputs['logits'][0, current_masked_index, :].log_softmax(dim=0)  # 计算概率

        values, predictions = probs.topk(self.beam_width)
        return zip(values.tolist(), predictions.tolist(), [self.tokenizer.decode(p) for p in predictions.tolist()])

    def __rerank_beam(self, beam_list):  # 补丁重排名
        reranked_beam_list = []
        for beam_index, zip_item in enumerate(beam_list):
            o_prob, tokens, codes = zip_item
            new_prob = 0
            for index in range(len(tokens)):
                new_input = self.input_ids.clone().detach()
                for j, token in enumerate(tokens):
                    if j != index:
                        new_input[0][self.masked_index[j]] = token
                outputs = self.model(input_ids=new_input, attention_mask=self.attention_mask)
                prob = outputs['logits'][0, self.masked_index[index], :].log_softmax(dim=0)[tokens[index]]
                new_prob += float(prob)

            reranked_beam_list.append((new_prob, tokens, codes))

        return reranked_beam_list

    # 生成初始波束列表，填充阵列，方便以后访问
    def __initial_beam_generate(self, current_masked_index):

        outputs = self.model(**self.inputs)
        probs = outputs['logits'][0, self.masked_index[current_masked_index], :].log_softmax(dim=0)
        values, predictions = probs.topk(self.beam_width)

        token_list = []
        code_list = []
        value_list = []
        for index, p in enumerate(predictions.tolist()):
            if "/" not in self.tokenizer.decode(p) and "@" not in self.tokenizer.decode(p) and "" != self.tokenizer.decode(p).strip():
                temp_token_l = [None] * len(self.masked_index)
                temp_token_l[current_masked_index] = p
                token_list.append(temp_token_l)
                temp_code_l = [None] * len(self.masked_index)
                temp_code_l[current_masked_index] = self.tokenizer.decode(p)
                code_list.append(temp_code_l)
                value_list.append(values.tolist()[index])

        beam_list = list(zip(value_list, token_list, code_list))
        return beam_list

    def generate_beam(self):
        current_masked_index = 0
        beam_list = self.__initial_beam_generate(current_masked_index)

        for index in self.masked_index[1:]:
            temp_beam_list = []
            current_masked_index += 1
            for item in beam_list:
                temp_zip = self.__beam_generate(item[1], index)
                for zip_item in temp_zip:
                    if "/" not in zip_item[2] and "@" not in zip_item[2] and "" != zip_item[2].strip():
                        # 数值稳定条件概率计算
                        temp_token_l = list(item[1])
                        temp_token_l[current_masked_index] = zip_item[1]
                        temp_code_l = list(item[2])
                        temp_code_l[current_masked_index] = zip_item[2]
                        temp_beam_list.append((item[0] + float(zip_item[0]), temp_token_l,
                                               temp_code_l))

            temp_beam_list.sort(key=lambda x: x[0], reverse=True)
            beam_list = [x for x in temp_beam_list[:self.beam_width]]
        if self.re_rank:
            beam_list = self.__rerank_beam(beam_list)

        return beam_list, self.masked_index


# Gibbs sampling method 吉布斯抽样法，并没有取得很好的效果
# 这里仅作比较，将来可能会有所改进
class GibbsBeamSearch(BeamSearch):

    def generate_beam(self):

        current_masked_index = random.randint(0, len(self.masked_index) - 1)
        beam_list = self.__initial_beam_generate(current_masked_index)

        for index in self.masked_index[1:]:
            masked_indices = [i for i, x in enumerate(beam_list[0][1]) if x is None]
            current_masked_index = random.choice(masked_indices)
            temp_beam_list = []
            for item in beam_list:
                temp_zip = self.__beam_generate(item[1], current_masked_index)
                for zip_item in temp_zip:
                    if "/" not in zip_item[2] and "" != zip_item[2].strip():
                        temp_token_l = list(item[1])
                        temp_token_l[current_masked_index] = zip_item[1]
                        temp_code_l = list(item[2])
                        temp_code_l[current_masked_index] = zip_item[2]
                        temp_beam_list.append((item[0] + zip_item[0], temp_token_l,
                                               temp_code_l))  # 数值稳定条件概率计算

            temp_beam_list.sort(key=lambda x: x[0], reverse=True)
            beam_list = [x for x in temp_beam_list[:self.beam_width]]

        if self.re_rank:
            beam_list = self.__rerank_beam(beam_list)

        return beam_list, self.masked_index
