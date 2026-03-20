

class RotInHEONGPU:
    def __init__(self,logSlots,StoC_piece, CtoS_piece):
        self.E_index_ = []
        self.E_size_ = []
        self.E_inv_index_ = []
        self.E_inv_size_ = []
        self.logSlots = logSlots
        self.numSlots = 2**logSlots
        self.StoC_piece_ = StoC_piece
        self.CtoS_piece_ = CtoS_piece
        self.E_splitted_ = []
        self.E_splitted_index_ = []
        self.V_matrixs_index_ = []
        self.E_splitted_diag_index_gpu_ = []
        self.E_splitted_iteration_gpu_ = []
        self.E_splitted_output_index_gpu_   = []
        self.E_splitted_input_index_gpu_ = []
    def unique_sort(self,input):
        result = set(input)
        return list(sorted(result))
    def generate_E_diagonals_index(self):
        first = True
        for i in range(1,self.logSlots+1):
            if first:
                block_size = self.numSlots // (2**i)
                self.E_index_.append(0)
                self.E_index_.append(block_size)
                self.E_size_.append(2)
                first = False
            else:
                block_size = self.numSlots // (2**i)
                self.E_index_.append(0)
                self.E_index_.append(block_size)
                self.E_index_.append(self.numSlots - block_size)
                self.E_size_.append(3)
    def generate_E_inv_diagonals_index(self):
        for i in range(self.logSlots, 0, -1):
            if i == 1:
                block_size = self.numSlots >> i
                self.E_inv_index_.append(0)
                self.E_inv_index_.append(block_size)
                self.E_inv_size_.append(2)
            else:
                block_size = self.numSlots >> i
                self.E_index_.append(0)
                self.E_inv_index_.append(block_size)
                self.E_inv_index_.append(self.numSlots - block_size)
                self.E_inv_size_.append(3)
    def split_E(self):
        k = self.logSlots // self.StoC_piece_
        m = self.logSlots % self.StoC_piece_

        for i in range(self.StoC_piece_):
            self.E_splitted_.append(k)

        for i in range(m):
            self.E_splitted_[i] += 1

        counter = 0
        for i in range(self.StoC_piece_):
            temp = []
            for j in range(self.E_splitted_[i]):
                size = 2 if counter == 0 else 3
                for k in range(size):
                    temp.append(self.E_index_[counter])
                    counter += 1
            self.E_splitted_index_.append(temp)

        num_slots_mask = self.numSlots - 1
        counter = 0
        for k in range(self.StoC_piece_):
            matrix_count = self.E_splitted_[k]
            L_m_loc = 2 if k == 0 else 3
            index_mul = []
            index_mul_sorted = []
            diag_index_temp = []
            iteration_temp = []
            for m in range(matrix_count - 1):
                if m == 0:
                    iteration_temp.append(self.E_size_[counter])
                    for i in range(self.E_size_[counter]):
                        R_m_INDEX = self.E_splitted_index_[k][i]
                        diag_index_temp.append(R_m_INDEX)
                        for j in range(self.E_size_[counter + 1]):
                            L_m_INDEX = self.E_splitted_index_[k][L_m_loc + j]
                            index_mul.append((L_m_INDEX + R_m_INDEX) & num_slots_mask)
                    index_mul_sorted = self.unique_sort(index_mul)
                    index_mul.clear()
                    L_m_loc += 3
                else:
                    iteration_temp.append(len(index_mul_sorted))
                    for i in range(len(index_mul_sorted)):
                        R_m_INDEX = index_mul_sorted[i]
                        diag_index_temp.append(R_m_INDEX)
                        for j in range(self.E_size_[counter + 1 + m]):
                            L_m_INDEX = self.E_splitted_index_[k][L_m_loc + j]
                            index_mul.append((L_m_INDEX + R_m_INDEX) & num_slots_mask)
                    index_mul_sorted = self.unique_sort(index_mul)
                    index_mul.clear()
                    L_m_loc += 3
            self.V_matrixs_index_.append(index_mul_sorted)
            self.E_splitted_diag_index_gpu_.append(diag_index_temp)
            self.E_splitted_iteration_gpu_.append(iteration_temp)
            counter += matrix_count

        dict_output_index = []
        for k in range(self.StoC_piece_):
            temp = {}
            for i in range(len(self.V_matrixs_index_[k])):
                temp[self.V_matrixs_index_[k][i]] = i
            dict_output_index.append(temp)

        counter = 0
        for k in range(self.StoC_piece_):
            matrix_count = self.E_splitted_[k]
            L_m_loc = 2 if k == 0 else 3
            index_mul = []
            index_mul_sorted = []

            temp_in_index = []
            temp_out_index = []
            for m in range(matrix_count - 1):
                if m == 0:
                    for i in range(self.E_size_[counter]):
                        R_m_INDEX = self.E_splitted_index_[k][i]
                        for j in range(self.E_size_[counter + 1]):
                            L_m_INDEX = self.E_splitted_index_[k][L_m_loc + j]
                            indexs = (L_m_INDEX + R_m_INDEX) & num_slots_mask
                            index_mul.append(indexs)
                            temp_out_index.append(
                                dict_output_index[k][indexs])
                    index_mul_sorted = self.unique_sort(index_mul)
                    index_mul.clear()
                    L_m_loc += 3
                else:
                    for i in range(len(index_mul_sorted)):
                        R_m_INDEX = index_mul_sorted[i]
                        temp_in_index.append(dict_output_index[k][R_m_INDEX])
                        for j in range(self.E_size_[counter + 1 + m]):
                            L_m_INDEX = self.E_splitted_index_[k][L_m_loc + j]
                            indexs = (L_m_INDEX + R_m_INDEX) & num_slots_mask
                            index_mul.append(indexs)
                            temp_out_index.append(dict_output_index[k][indexs])
                    index_mul_sorted = self.unique_sort(index_mul)
                    index_mul.clear()
                    L_m_loc += 3
            counter += matrix_count
            self.E_splitted_input_index_gpu_.append(temp_in_index)
            self.E_splitted_output_index_gpu_.append(temp_out_index)
            
a = RotInHEONGPU(11,4,4)
a.generate_E_diagonals_index()
a.split_E()
print(a)
