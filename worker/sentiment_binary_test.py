import os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from utill.SNS_content_text_one_module_for_senti_binary import SnsContentClassifier
import timeit
import pandas as pd
import numpy as np

def Accuracy(df):
    true_y = np.array(df['label'])
    predict_y = np.array(df['y_hat'])
    accuracy = np.mean(np.equal(true_y, predict_y))
    return accuracy

def Precision(df):
    true_y = np.array(df['label'])
    predict_y = np.array(df['y_hat'])
    right = np.sum(true_y*predict_y == 1)
    precision = right/np.sum(predict_y)
    return precision

def Recall(df):
    true_y = np.array(df['label'])
    predict_y = np.array(df['y_hat'])
    right = np.sum(true_y*predict_y == 1)
    recall = right/np.sum(true_y)
    return recall

if __name__=='__main__':
    start_time = timeit.default_timer()
    ###############오늘 SNS contents 데이터 로드##############################################
    SNS_content_df_whole = pd.DataFrame(columns=['document','label','label2'])
    # test van
    # for i in range(10):
    for i in range(2):
        i = i+1
        try:

            SNS_content_df = pd.read_csv(os.path.join('..','input','sentiment_test_100_%d.csv') %(i))
        except FileNotFoundError:
            SNS_content_df = pd.read_csv(os.path.join(os.getcwd(), 'input', 'sentiment_test_100_%d.csv') %(i))
        SNS_content_df = SNS_content_df[['document','label','label2']]
        # add van, concat(axis=0) : DataFrame을 아래로 계속 합치기
        SNS_content_df_whole = pd.concat([SNS_content_df_whole, SNS_content_df],axis=0)
    SNS_content_df_whole.reset_index(drop=True, inplace=True)
    try:
        SNS_content_df_whole.to_csv(os.path.join('..', 'input', 'sentiment_test_input_whole.csv'), mode='w', encoding='utf-8-sig')
    except FileNotFoundError:
        SNS_content_df_whole.to_csv(os.path.join(os.getcwd(), 'input', 'sentiment_test_input_whole.csv'), mode='w',
                                    encoding='utf-8-sig')
    ################랜덤 샘플링 & 콘텐츠 분류, 저장, 정확도 측정##############################################################
    SNS_contents_classifier = SnsContentClassifier()
    Accuracy_list = []
    Precision_list = []
    Recall_list = []
    # test van
    #for i in range(10):
    for i in range(2):
        globals()["SNS_content_df_{}".format(i)] = SNS_content_df_whole.sample(100, random_state=2019)
        result = SNS_contents_classifier.run(globals()["SNS_content_df_{}".format(i)])
        ## 각각 정확도 측정
        globals()["SNS_content_df_{}_Accuracy".format(i)] = Accuracy(result)*100
        globals()["SNS_content_df_{}_Precision".format(i)] = Precision(result)*100
        globals()["SNS_content_df_{}_Recall".format(i)] = Recall(result)*100
        Accuracy_list.append(globals()["SNS_content_df_{}_Accuracy".format(i)])
        Precision_list.append(globals()["SNS_content_df_{}_Precision".format(i)])
        Recall_list.append(globals()["SNS_content_df_{}_Recall".format(i)])
        result = result[['document', 'label2', 'y_hat_label']]
        result.columns = ['document', '실제긍부정','예측긍부정']
        try:
            result.to_csv(os.path.join(os.getcwd(), 'output', 'SNS_content_df_{}.csv'.format(i)), mode='w', encoding='utf-8-sig')
        except FileNotFoundError:
            result.to_csv(
                os.path.join('..', 'output', 'SNS_content_df_{}.csv'.format(i)), mode='w', encoding='utf-8-sig')
        SNS_content_df_whole.drop(globals()["SNS_content_df_{}".format(i)].index, inplace=True)

    # test van
    #for i in range(10):
    for i in range(2):
        print("SNS_content_df_{}_Accuracy: {}%".format(i, globals()["SNS_content_df_{}_Accuracy".format(i)]))
        print("SNS_content_df_{}_Precision: {}%".format(i, globals()["SNS_content_df_{}_Precision".format(i)]))
        print("SNS_content_df_{}_Recall: {}%".format(i, globals()["SNS_content_df_{}_Recall".format(i)]))

    #################최종 성능 지표   ########################################################
    print("총 평균 Accuracy는 {}% 입니다".format(sum(Accuracy_list)/len(Accuracy_list)))
    print("총 평균 Precision은 {}% 입니다".format(sum(Precision_list) / len(Precision_list)))
    print("총 평균 Recall은 {}% 입니다".format(sum(Recall_list) / len(Recall_list)))
    terminate_time = timeit.default_timer() # 종료 시간 체크
    print("%f초 걸렸습니다." % (terminate_time - start_time))
    ################SNS contents 분류 #######################################################

    #
    # try:
    #     result.to_csv(os.path.join(os.getcwd(), 'ouput', 'sentiment_binary_test_result.csv'), mode='w', encoding='utf-8-sig')
    # except FileNotFoundError:
    #     result.to_csv(os.path.join('..','ouput','sentiment_binary_test_result.csv'),mode='w',encoding='utf-8-sig')

