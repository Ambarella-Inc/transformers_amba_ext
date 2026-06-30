import ctypes
import enum

MAX_FILENAME_LEN = 128
MAX_RANK_NUM = 32
MAX_RANK_ADDR_LEN = 32

"""
/*!
 * @brief The log level dist_ai library
 */
typedef enum log_e {
	LOG_ERR = 0,            /*!< Log level error */
	LOG_WARN,               /*!< Log level warn */
	LOG_INFO,               /*!< Log level info */
	LOG_DEBUG,              /*!< Log level debug */
	LOG_VERBOSE,            /*!< Log level verbose */
	LOG_DEFAULT = LOG_INFO, /*!< Log level default */
} log_t;
"""
class log_t(enum.IntEnum):
	LOG_ERR = 0
	LOG_WARN = 1
	LOG_INFO = 2
	LOG_DEBUG = 3
	LOG_VERBOSE = 4
	LOG_DEFAULT = LOG_INFO

"""
/*!
 * @brief The version of dist_ai library
 */
struct dist_ai_version {
	uint32_t major;       /*!< Version major number */
	uint32_t minor;       /*!< Version minor number */
	uint32_t patch;       /*!< Version patch number */
	uint32_t mod_time;    /*!< Version modification time */
	char description[64]; /*!< Version description */
};
"""
class dist_ai_version(ctypes.Structure):
	_fields_ = [
		("major", ctypes.c_uint32),
		("minor", ctypes.c_uint32),
		("patch", ctypes.c_uint32),
		("mod_time", ctypes.c_uint32),
		("description", ctypes.c_char * 64),
	]

"""
typedef struct tokenizer_enc_cfg_s {
	uint32_t add_special_token;
	uint32_t reserved[31];
} tokenizer_enc_cfg_t;
"""

"""
 typedef struct tokenizer_token_s {
	uint32_t *token;
	uint32_t num;
	uint32_t max_num;
	uint32_t reserved[28];
} tokenizer_token_t;
"""

"""
typedef struct tokenizer_dec_cfg_s {
	uint32_t raw_word;
	uint32_t strip_flag;
	uint32_t skip_special_tokens;
	uint32_t reserved[29];
} tokenizer_dec_cfg_t;
"""

"""
typedef struct tokenizer_text_s {
	char *text;
	uint32_t length;
	uint32_t max_length;
	uint32_t reserved[28];
} tokenizer_text_t;
"""

"""
typedef enum dist_ai_infer_mode_e {
	INF_MODE_NORMAL = 0,
	INF_MODE_DEBUG = 1,
} dist_ai_infer_mode_t;
"""
class dist_ai_infer_mode_t(enum.IntEnum):
	INF_MODE_NORMAL = 0
	INF_MODE_DEBUG = 1

"""
typedef enum dist_ai_submod_en_e {
	SUBM_NONE = 0,

	/* enable sub module flag */
	SUBM_EN_MPI = BIT(0),
	SUBM_EN_ATTN = BIT(1),
	SUBM_EN_MOE = BIT(2),
	SUBM_EN_LM_HEAD = BIT(3),
} dist_ai_submod_en_t;
"""
class dist_ai_submod_en_t(enum.IntEnum):
	SUBM_NONE = 0
	SUBM_EN_MPI = 1 << 0
	SUBM_EN_ATTN = 1 << 1
	SUBM_EN_MOE = 1 << 2
	SUBM_EN_LM_HEAD = 1 << 3

"""
typedef struct dist_ai_ext_config_s {
	uint32_t log_level : 8;
	uint32_t reserved_0 : 24;
	uint32_t rank_id;
	uint32_t rank_root_id;
	uint32_t rank_num;
	dist_ai_submod_en_t submod_en_bitmap;
	OUT uint32_t max_context_len; // return value
	OUT uint32_t eos_token_id; // return value
	uint32_t reserved[25];
	char model_path[MAX_FILENAME_LEN];
	char rank_addr_table[MAX_RANK_NUM][MAX_RANK_ADDR_LEN];
} dist_ai_ext_config_t;
"""
class dist_ai_ext_config_t(ctypes.Structure):
	_fields_ = [
		("log_level", ctypes.c_uint32, 8),
		("reserved_0", ctypes.c_uint32, 24),
		("rank_id", ctypes.c_uint32),
		("rank_root_id", ctypes.c_uint32),
		("rank_num", ctypes.c_uint32),
		("submod_en_bitmap", ctypes.c_int),
		("max_context_len", ctypes.c_uint32),
		("eos_token_id", ctypes.c_uint32),
		("reserved", ctypes.c_uint32 * 25),
		("model_path", ctypes.c_char * MAX_FILENAME_LEN),
		("rank_addr_table", (ctypes.c_char * MAX_RANK_ADDR_LEN) * MAX_RANK_NUM),
	]

"""
typedef struct dist_ai_run_config_s {
	float temp;
	uint32_t topp;
	uint32_t topk;
	dist_ai_infer_mode_t infer_mode;
	uint32_t debug_bitmap;
	uint32_t reserved[3];
} dist_ai_run_config_t;
"""
class dist_ai_run_config_t(ctypes.Structure):
	_fields_ = [
		("temp", ctypes.c_float),
		("topp", ctypes.c_uint32),
		("topk", ctypes.c_uint32),
		("infer_mode", ctypes.c_int),
		("debug_bitmap", ctypes.c_uint32),
		("reserved", ctypes.c_uint32 * 3),
	]

"""
typedef struct dist_ai_perf_time_s {
	unsigned long prefill_time;
	unsigned long decode_time;

	unsigned long attn_time;
	unsigned long moe_time;
	unsigned long lm_head_time;
} dist_ai_perf_time_t;
"""
class dist_ai_perf_time_t(ctypes.Structure):
	_fields_ = [
		("prefill_time", ctypes.c_ulong),
		("decode_time", ctypes.c_ulong),
		("attn_time", ctypes.c_ulong),
		("moe_time", ctypes.c_ulong),
		("lm_head_time", ctypes.c_ulong),
	]

"""
typedef struct dist_ai_out_s {
	uint32_t pos;
	uint32_t token;
	uint32_t is_eos : 1;
	uint32_t reserved_0 : 31;
	struct dist_ai_perf_time_s time;
} dist_ai_out_t;
"""
class dist_ai_out_t(ctypes.Structure):
	_fields_ = [
		("pos", ctypes.c_uint32),
		("token", ctypes.c_uint32),
		("is_eos", ctypes.c_uint32, 1),
		("reserved_0", ctypes.c_uint32, 31),
		("time", dist_ai_perf_time_t),
	]

"""
typedef enum dist_ai_reset_type_e {
	RESET_TYPE_HARD = 0, /*!< Reset position to zero and clean last turn conversation */
	RESET_TYPE_SOFT,     /*!< Reset position to soft reset position
		* and keep last turn conversation */
	RESET_TYPE_ROLLBACK, /*!< Rollback position to a history place,
		* typically, the end of last turn conversation */
	RESET_TYPE_FIRST = RESET_TYPE_HARD,
	RESTE_TYPE_LAST = RESET_TYPE_ROLLBACK,
} dist_ai_reset_type_t;
"""
class dist_ai_reset_type_t(enum.IntEnum):
	RESET_TYPE_HARD = 0
	RESET_TYPE_SOFT = 1
	RESET_TYPE_ROLLBACK = 2
	RESET_TYPE_FIRST = RESET_TYPE_HARD
	RESTE_TYPE_LAST = RESET_TYPE_ROLLBACK

"""
typedef struct dist_ai_reset_cfg_s {
	dist_ai_reset_type_t reset_type; /*!< Reset type */
	uint32_t reset_pos;   /*!< Input, user set context to specify pos */
	uint32_t roll_back_pos;/*!< Rollback postion in inference context window */
	uint32_t reserved[61]; /*!< Reserved field */
} dist_ai_reset_cfg_t;
"""
class dist_ai_reset_cfg_t(ctypes.Structure):
	_fields_ = [
		("reset_type", ctypes.c_int),
		("reset_pos", ctypes.c_uint32),
		("roll_back_pos", ctypes.c_uint32),
		("reserved", ctypes.c_uint32 * 61),
	]

"""
typedef struct dist_ai_bcast_s {
	uint32_t ids_num;
	uint32_t is_reset : 1;
	uint32_t is_eot : 1;
	uint32_t reserved_0 : 30;
	dist_ai_reset_type_t reset_type;
	uint32_t reset_pos;
	uint32_t reserved[4];
} dist_ai_bcast_t;
"""
class dist_ai_bcast_t(ctypes.Structure):
	_fields_ = [
		("ids_num", ctypes.c_uint32),
		("is_reset", ctypes.c_uint32, 1),
		("is_eot", ctypes.c_uint32, 1),
		("reserved_0", ctypes.c_uint32, 30),
		("reset_type", ctypes.c_int),
		("reset_pos", ctypes.c_uint32),
		("reserved", ctypes.c_uint32 * 4),
	]

"""
typedef struct dist_ai_mpi_bm_cfg_s {
	uint32_t warmup_num;
	uint32_t itera_num;
	uint32_t extra_itera_num;
	uint32_t min_msg_size;
	uint32_t max_msg_size;
	uint32_t crc32_check : 1;
	uint32_t dump_flag : 1;
	uint32_t mem_use_cache : 1;
	uint32_t reserved : 29;
	uint32_t mem_type;
} dist_ai_mpi_bm_cfg_t;
"""