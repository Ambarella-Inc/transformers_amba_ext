import numpy as np


class user_context():
	def __init__(self, user_hnd, user_id: np.uint32):
		self.id = user_id
		self.pos = 0
		self.handle = user_hnd
		self.vit_img_data = None

class infer_multi_user_ctx():
	def __init__(self, max_user_num: np.uint32):
		self.user_id_base = 1000
		self.cur_user_id = self.user_id_base
		self.max_user_num = max_user_num
		self.multi_user_list = []

	def __get_user_ctx_idx(self, user_id):
		if user_id < self.user_id_base or user_id > self.cur_user_id:
			raise ValueError(f"Invalid user_id: {user_id}, "
				"should be [{self.user_id_base}, {self.cur_user_id}]")

		index = [index for index, _user_ctx in enumerate(self.multi_user_list) \
			if _user_ctx.id == user_id][0]
		return index

	def creat_user(self, user_handle):
		if self.get_user_cnt() >= self.max_user_num:
			raise ValueError(f"current user num: {self.get_user_cnt()} overflow, "
				"Please set max_user_num for model config")
		user_ctx = user_context(user_handle, self.cur_user_id)
		self.multi_user_list.append(user_ctx)
		self.cur_user_id += 1
		return user_ctx

	def release_user(self, user_id):
		user_index = self.__get_user_ctx_idx(user_id)
		self.multi_user_list.pop(user_index)

	def release_all_user(self):
		self.multi_user_list.clear()

	def get_user_ctx_with_index(self, user_index):
		return self.multi_user_list[user_index]

	def get_user_ctx(self, user_id):
		user_index = self.__get_user_ctx_idx(user_id)
		return self.multi_user_list[user_index]

	def get_user_cnt(self):
		return len(self.multi_user_list)

	def get_first_user_ctx(self):
		return self.multi_user_list[0]

	def update_user_pos(self, user_id, user_pos):
		user_index = self.__get_user_ctx_idx(user_id)
		self.multi_user_list[user_index].pos = user_pos

	def update_user_img_data(self, user_id, img_data):
		user_index = self.__get_user_ctx_idx(user_id)
		self.multi_user_list[user_index].vit_img_data = img_data
		# user_ctx = self.multi_user_list[user_index]
		# print(f"update_user_img_data: {user_ctx.id}, {user_ctx.pos}, {user_ctx.handle}, {user_ctx.vit_img_data}")

	def show_user_list(self):
		for _user_ctx in self.multi_user_list:
			print(f"user id: {_user_ctx.id}, handle: {_user_ctx.handle}, pos: {_user_ctx.pos}")
