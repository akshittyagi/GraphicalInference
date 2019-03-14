import numpy as np

class DAG(object):

    def __init__(self, filepath):
        '''@init docstring'''
        self.file = open(filepath, 'r')
        self.map = {'A':0, 'G':1, 'CP':2, 'BP':3, 'CH':4, 'ECG':5, 'HR':6, 'EIA':7, 'HD':8}
        self.var_space = {'A':3, 'G':2, 'CP':4, 'BP':2, 'CH':2, 'ECG':2, 'HR':2, 'EIA':2, 'HD':2}
        self.data = []
        self.np_data = []
        self.N = 0

    def read_data(self):
        '''@read_data docstring'''
        for line in self.file:
            nums = line.split(',')
            nums = [int(elem) for elem in nums]
            self.data.append(nums)
        self.np_data = np.array(self.data)
        self.N = self.np_data.shape[0]
    
    def estimate_proba_A(self):
        '''@estimate_proba docstring'''
        count_1 = np.sum((self.np_data[:, self.map['A']]) == 1)
        count_2 = np.sum((self.np_data[:, self.map['A']]) == 2)
        count_3 = self.N - count_1 - count_2
        print("p(A = <45): ", count_1*1.0/self.N)
        print("p(A = 45-55): ", count_2*1.0/self.N)
        print("p(A = >55): ", count_3*1.0/self.N)

    def estimate_proba_BP_G(self):
        '''@estimate_proba docstring'''
        low = (self.np_data[:, self.map['BP']] == 1)
        high = (self.np_data[:, self.map['BP']] == 2)
        fem = (self.np_data[:, self.map['G']] == 1)
        male = (self.np_data[:, self.map['G']] == 2)
        count_low_fem = np.sum(low & fem)
        count_high_fem = np.sum(high & fem)
        count_low_male = np.sum(low & male)
        count_high_male = np.sum(high & male)
        fem = np.sum(fem)
        male = np.sum(male)
        print("p(BP=low|G=female): ", count_low_fem*1.0/(fem)) 
        print("p(BP=high|G=female): ", count_high_fem*1.0/(fem)) 
        print("p(BP=low|G=male): ", count_low_male*1.0/(male)) 
        print("p(BP=high|G=male): ", count_high_male*1.0/(male)) 

    def estimate_proba_HD_BP_CH(self):
        '''@estimate_proba docstring'''
        low = (self.np_data[:, self.map['BP']] == 1)
        high = (self.np_data[:, self.map['BP']] == 2)
        no = (self.np_data[:, self.map['HD']] == 1)
        yes = (self.np_data[:, self.map['HD']] == 2)
        l = (self.np_data[:, self.map['CH']] == 1)
        h = (self.np_data[:, self.map['CH']] == 2)
        count_no_low_l = np.sum((low & l) & no)
        count_yes_low_l = np.sum((low & l) & yes)
        count_no_high_l = np.sum((high & l) & no)
        count_yes_high_l = np.sum((high & l) & yes)
        count_no_low_h = np.sum((low & h) & no)
        count_yes_low_h = np.sum((low & h) & yes)
        count_no_high_h = np.sum((high & h) & no)
        count_yes_high_h = np.sum((high & h) & yes)
        low_l = np.sum(low & l)
        low_h = np.sum(low & h)
        high_l = np.sum(high & l)
        high_h = np.sum(high & h)
        print("p(HD=no|BP=low,CH=low): ", count_no_low_l*1.0/(low_l))
        print("p(HD=yes|BP=low,CH=low): ", count_yes_low_l*1.0/(low_l))
        print("p(HD=no|BP=high,CH=low): ", count_no_high_l*1.0/(high_l))
        print("p(HD=yes|BP=high,CH=low): ", count_yes_high_l*1.0/(high_l))
        print("p(HD=no|BP=low,CH=high): ", count_no_low_h*1.0/(low_h))
        print("p(HD=yes|BP=low,CH=high): ", count_yes_low_h*1.0/(low_h))
        print("p(HD=no|BP=high,CH=high): ", count_no_high_h*1.0/(high_h))
        print("p(HD=yes|BP=high,CH=high): ", count_yes_high_h*1.0/(high_h))
    
    def estimate_proba_HR_A_HD(self):
        '''@estimate_proba docstring'''
        low = (self.np_data[:, self.map['HR']] == 1)
        high = (self.np_data[:, self.map['HR']] == 2)
        no = (self.np_data[:, self.map['HD']] == 1)
        yes = (self.np_data[:, self.map['HD']] == 2)
        l = (self.np_data[:, self.map['A']] == 1)
        m = (self.np_data[:, self.map['A']] == 2)
        h = (self.np_data[:, self.map['A']] == 3)
        count_no_low_l = np.sum((low & l) & no)
        count_yes_low_l = np.sum((low & l) & yes)
        count_no_high_l = np.sum((high & l) & no)
        count_yes_high_l = np.sum((high & l) & yes)
        count_no_low_h = np.sum((low & h) & no)
        count_yes_low_h = np.sum((low & h) & yes)
        count_no_high_h = np.sum((high & h) & no)
        count_yes_high_h = np.sum((high & h) & yes)
        count_no_low_m = np.sum((low & m) & no)
        count_yes_low_m = np.sum((low & m) & yes)
        count_no_high_m = np.sum((high & m) & no)
        count_yes_high_m = np.sum((high & m) & yes)
        no_l = np.sum(no & l)
        no_h = np.sum(no & h)
        no_m = np.sum(no & m)
        yes_l = np.sum(yes & l)
        yes_h = np.sum(yes & h)
        yes_m = np.sum(yes & m)
        print("p(HR=low|HD=no, A=low): ", count_no_low_l*1.0/no_l)
        print("p(HR=high|HD=no, A=low): ", count_no_high_l*1.0/no_l)
        print("p(HR=low|HD=yes, A=low): ", count_yes_low_l*1.0/yes_l)
        print("p(HR=high|HD=yes, A=low): ", count_yes_high_l*1.0/yes_l)
        print("p(HR=low|HD=no, A=med): ", count_no_low_m*1.0/no_m)
        print("p(HR=high|HD=no, A=med): ", count_no_high_m*1.0/no_m)
        print("p(HR=low|HD=yes, A=med): ", count_yes_low_m*1.0/yes_m)
        print("p(HR=high|HD=yes, A=med): ", count_yes_high_m*1.0/yes_m)
        print("p(HR=low|HD=no, A=high): ", count_no_low_h*1.0/no_h)
        print("p(HR=high|HD=no, A=high): ", count_no_high_h*1.0/no_h)
        print("p(HR=low|HD=yes, A=high): ", count_yes_low_h*1.0/yes_h)
        print("p(HR=high|HD=yes, A=high): ", count_yes_high_h*1.0/yes_h)
    
    def estimate_proba_CH_G_A(self):
        '''@estimate_proba docstring'''
        low = (self.np_data[:, self.map['CH']] == 1)
        high = (self.np_data[:, self.map['CH']] == 2)
        fem = (self.np_data[:, self.map['G']] == 1)
        male = (self.np_data[:, self.map['G']] == 2)
        l = (self.np_data[:, self.map['A']] == 1)
        m = (self.np_data[:, self.map['A']] == 2)
        h = (self.np_data[:, self.map['A']] == 3)
        count_fem_low_l = np.sum((low & l) & fem)
        count_male_low_l = np.sum((low & l) & male)
        count_fem_high_l = np.sum((high & l) & fem)
        count_male_high_l = np.sum((high & l) & male)
        count_fem_low_h = np.sum((low & h) & fem)
        count_male_low_h = np.sum((low & h) & male)
        count_fem_high_h = np.sum((high & h) & fem)
        count_male_high_h = np.sum((high & h) & male)
        count_fem_low_m = np.sum((low & m) & fem)
        count_male_low_m = np.sum((low & m) & male)
        count_fem_high_m = np.sum((high & m) & fem)
        count_male_high_m = np.sum((high & m) & male)
        fem_l = np.sum(fem & l)
        fem_h = np.sum(fem & h)
        fem_m = np.sum(fem & m)
        male_l = np.sum(male & l)
        male_h = np.sum(male & h)
        male_m = np.sum(male & m)
        print("p(CH=low|G=fem, A=low): ", count_fem_low_l*1.0/fem_l)
        print("p(CH=high|G=fem, A=low): ", count_fem_high_l*1.0/fem_l)
        print("p(CH=low|G=male, A=low): ", count_male_low_l*1.0/male_l)
        print("p(CH=high|G=male, A=low): ", count_male_high_l*1.0/male_l)
        print("p(CH=low|G=fem, A=med): ", count_fem_low_m*1.0/fem_m)
        print("p(CH=high|G=fem, A=med): ", count_fem_high_m*1.0/fem_m)
        print("p(CH=low|G=male, A=med): ", count_male_low_m*1.0/male_m)
        print("p(CH=high|G=male, A=med): ", count_male_high_m*1.0/male_m)
        print("p(CH=low|G=fem, A=high): ", count_fem_low_h*1.0/fem_h)
        print("p(CH=high|G=fem, A=high): ", count_fem_high_h*1.0/fem_h)
        print("p(CH=low|G=male, A=high): ", count_male_low_h*1.0/male_h)
        print("p(CH=high|G=male, A=high): ", count_male_high_h*1.0/male_h)
    
    def query_from_data(self, query_str):
        queries = query_str.split('|')
        marginal = queries[0].split("=")
        marginal_var = self.map[marginal[0]]
        marginal_val = int(marginal[1])
        marg = self.np_data[:,marginal_var] == marginal_val
        if len(queries) > 1:
            conditional = queries[1:]    
            conditionals = conditional[0].split(",")
            conditionals = [(self.map[elem.split("=")[0]], int(elem.split("=")[1])) for elem in conditionals]
            first_conditional = self.np_data[:, conditionals[0][0]] == conditionals[0][1]
            total = marg & first_conditional
            denominator = first_conditional
            for i in range(1, len(conditionals)):
                current_conditional = self.np_data[:, conditionals[i][0]] == conditionals[i][1]
                total = total & current_conditional
                denominator = denominator & current_conditional
            return np.sum(total)*1.0/np.sum(denominator)
        else:
            return np.sum(marg)*1.0/self.N

    def estimate_proba_CH_rest(self):
        numerator_low =  self.query_from_data('HD=1|BP=1,CH=1') * self.query_from_data('CH=1|G=2,A=2')
        numerator_high = self.query_from_data('HD=1|BP=1,CH=2') * self.query_from_data('CH=2|G=2,A=2')
        denominator = numerator_high + numerator_low
        print("p(CH=low|Rest): ", numerator_low*1.0/denominator)
        print("p(CH=high|Rest): ", numerator_high*1.0/denominator)
    
    def estimate_proba_BP_rest_sans_G(self):
        numerator_low = self.query_from_data('HD=1|BP=1,CH=2') * self.query_from_data('BP=1|G=1') * self.query_from_data('CH=2|G=1,A=2') * self.query_from_data('G=1')
        numerator_low += self.query_from_data('HD=1|BP=1,CH=2') * self.query_from_data('BP=1|G=2') * self.query_from_data('CH=2|G=2,A=2') * self.query_from_data('G=2')
        numerator_high = self.query_from_data('HD=1|BP=2,CH=2') * self.query_from_data('BP=2|G=1') * self.query_from_data('CH=2|G=1,A=2') * self.query_from_data('G=1')
        numerator_high += self.query_from_data('HD=1|BP=2,CH=2') * self.query_from_data('BP=2|G=2') * self.query_from_data('CH=2|G=2,A=2') * self.query_from_data('G=2')
        denominator = numerator_high + numerator_low
        print("p(BP=low|Rest-G): ", numerator_low*1.0/denominator)
        print("p(BP=high|Rest-G): ", numerator_high*1.0/denominator)

    def validation_testing(self):
        train_pre = '../Data/data-train-'
        test_pre = '../Data/data-test-'
        post = '.txt'
        data = [(train_pre+str(i)+post,test_pre+str(i)+post) for i in range(1, 6)]
        
        accuracies = []
        for idx, val in enumerate(data):
            train_file = val[0]
            test_file = val[1]
            new_dag = DAG(train_file)
            new_dag.read_data()
            fil = open(test_file, 'r')
            counts = 0
            total = 0
            for line in fil:
                line.strip("\r\n")
                total += 1
                print("TESTING FOR: ", test_file, " AT: ", total)
                curr_data_point = line.split(",")
                cp = curr_data_point[self.map['CP']]
                eia = curr_data_point[self.map['EIA']]
                ecg = curr_data_point[self.map['ECG']]
                hr = curr_data_point[self.map['HR']]
                a = curr_data_point[self.map['A']]
                bp = curr_data_point[self.map['BP']]
                ch = curr_data_point[self.map['CH']]
                numerator_low = new_dag.query_from_data('CP='+cp+'|HD=1') * new_dag.query_from_data("EIA="+eia+"|HD=1") * new_dag.query_from_data("ECG="+ecg+"|HD=1") * new_dag.query_from_data("HR="+hr+"|HD=1,A="+a) * new_dag.query_from_data("HD=1|BP="+bp+",CH="+ch)
                numerator_high = new_dag.query_from_data('CP='+cp+"|HD=2") * new_dag.query_from_data("EIA="+eia+"|HD=2") * new_dag.query_from_data("ECG="+ecg+"|HD=2") * new_dag.query_from_data("HR="+hr+"|HD=2,A="+a) * new_dag.query_from_data("HD=2|BP="+bp+",CH="+ch)
                prediction = (1 if numerator_low > numerator_high else 2)
                if prediction == int(curr_data_point[self.map['HD']]):
                    counts += 1
            print("ACC: ", counts*1.0/(total))
            accuracies.append(counts*1.0/(total))
        print("ACCURACIES: ", accuracies)
        print("MEAN: ", np.sum(accuracies)*1.0/len(accuracies))
        accuracies = np.array(accuracies)
        accuracies -= np.sum(accuracies)*1.0/len(accuracies)
        stddev = np.sqrt(np.sum(accuracies**2)/len(accuracies))
        print("STDDEV: ", stddev)
    
    def validation_testing_new_model(self):
        train_pre = '../Data/data-train-'
        test_pre = '../Data/data-test-'
        post = '.txt'
        data = [(train_pre+str(i)+post,test_pre+str(i)+post) for i in range(1, 6)]
        
        accuracies = []
        for idx, val in enumerate(data):
            train_file = val[0]
            test_file = val[1]
            new_dag = DAG(train_file)
            new_dag.read_data()
            fil = open(test_file, 'r')
            counts = 0
            total = 0
            for line in fil:
                line.strip("\r\n")
                total += 1
                print("TESTING FOR: ", test_file, " AT: ", total)
                curr_data_point = line.split(",")
                ecg = curr_data_point[self.map['ECG']]
                bp = curr_data_point[self.map['BP']]
                ch = curr_data_point[self.map['CH']]
                numerator_low = new_dag.query_from_data('BP='+bp+'|HD=1') * new_dag.query_from_data("HD=1|ECG="+ecg+",CH="+ch) 
                numerator_high = new_dag.query_from_data('BP='+bp+'|HD=2') * new_dag.query_from_data("HD=2|ECG="+ecg+",CH="+ch) 
                prediction = (1 if numerator_low > numerator_high else 2)
                if prediction == int(curr_data_point[self.map['HD']]):
                    counts += 1
            print("ACC: ", counts*1.0/(total))
            accuracies.append(counts*1.0/(total))
        print("ACCURACIES: ", accuracies)
        print("MEAN: ", np.sum(accuracies)*1.0/len(accuracies))
        accuracies = np.array(accuracies)
        accuracies -= np.sum(accuracies)*1.0/len(accuracies)
        stddev = np.sqrt(np.sum(accuracies**2)/len(accuracies))
        print("STDDEV: ", stddev)


if __name__ == "__main__":
    dag = DAG('../Data/data-train-1.txt')
    dag.read_data()
    part = 7
    if part == 4:
        dag.estimate_proba_A()
        dag.estimate_proba_BP_G()
        dag.estimate_proba_HD_BP_CH()
        dag.estimate_proba_HR_A_HD()
    if part == 5:
        dag.estimate_proba_CH_rest()
        dag.estimate_proba_BP_rest_sans_G()
    if part == 6:
        dag.validation_testing()
    if part == 7:
        dag.validation_testing_new_model()