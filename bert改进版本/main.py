import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM

from passage_process import pass_proce,ques_proce,ans_proce




print(1)



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertForMaskedLM.from_pretrained('bert-base-uncased')
bert.eval()
#bert.to('cuda:1')






#选项提取
choices=ques_proce('question.txt')
choices_idx=[]
for choice in choices:
    choice_idx=tokenizer.convert_tokens_to_ids(choice)
    choices_idx.append(choice_idx)





print('建立预测概率矩阵')
ans_prob=[]
for i in range(len(choices)):
    ans_prob.append([0.0,0.0,0.0,0.0])




print('文本处理')
text=pass_proce("passage.txt",10)
for mask_sen in text:
    for per_sen in mask_sen:
        tokenized_text = tokenizer.tokenize(per_sen[0])
        broke_point=tokenized_text.index('[SEP]')
        segments_ids=[0]*(broke_point+1)+[1]*(len(tokenized_text)-broke_point-1)
        que_idxs=per_sen[1]

        ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokenized_text)])
        segments_tensors = torch.tensor([segments_ids])
        #ids = ids.to('cuda:0')
        #segments_tensors = segments_tensors.to('cuda:0')


        #mask的位置提取
        mask_num=tokenized_text.count('[MASK]')
        mask_idxs=[idx for idx in range(len(tokenized_text)) if tokenized_text[idx]=='[MASK]']





        #预测答案


        result = bert(ids,segments_tensors)
        for i in range(mask_num):
            mask_idx=mask_idxs[i]
            this_ans_prob = [result[0][mask_idx][choice_idx] for choice_idx in choices_idx[que_idxs[i]]]
            ans_prob[que_idxs[i]]=[ans_prob[que_idxs[i]][j]+this_ans_prob[j] for j in range(4)]


#归一化
for i in range(len(choices)):
    for j in range(4):
        ans_prob[i][j]/=10





#计算预测答案
print(ans_prob)


ans_pred=[]
for per_que in ans_prob:
    max=0
    for i in range(4):
        if (per_que[i].item()>max):
            max=per_que[i].item()
            index=i



    ans=['A','B','C','D'][index]
    ans_pred.append(ans)

print(ans_pred)




#导入正确答案
ans_conrrect=ans_proce('answer.txt')



#计算正确率
correct=0.0
for i in range(len(choices)):
    if ans_pred[i]==ans_conrrect[i]:
        correct+=1
print("the correct rate is :"+str(correct/len(choices)*100.0)+"%")
