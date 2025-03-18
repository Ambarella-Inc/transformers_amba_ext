import ctypes
import enum

"""
typedef enum shepd_device_type_e {
	SHEPD_DEVICE_LOCAL = 0,  /*!< Local device, default option */
	SHEPD_DEVICE_REMOTE,     /*!< Remote device via PCIe or Ethernet */
} shepd_device_type_t;
"""
class shepd_device_type_t(enum.IntEnum):
	SHEPD_DEVICE_LOCAL = 0,
	SHEPD_DEVICE_REMOTE = 1
"""
struct shepd_device_cfg {
	IN shepd_device_type_t device_type; /*!< Device type */
	IN int device_port;                 /*!< Device port ID */
	IN char *device_ip;                 /*!< Device IP */
	IN uint32_t reserved[28];           /*!< Reserved field */
};
"""
class shepd_device_cfg(ctypes.Structure):
	_fields_ = [
		("device_type", ctypes.c_int),
		("device_port", ctypes.c_int),
		("device_ip", ctypes.c_char_p),
		("reserved", ctypes.c_uint32 * 28),
	]

"""
typedef enum shepd_reset_type_e {
	RESET_TYPE_HARD = 0, /*!< Reset position to zero and clean last turn conversation */
	RESET_TYPE_SOFT,     /*!< Reset position to soft reset position
		* and keep last turn conversation */
	RESET_TYPE_ROLLBACK, /*!< Rollback position to a history place,
		* typically, the end of last turn conversation */
	RESET_TYPE_FIRST = RESET_TYPE_HARD,
	RESTE_TYPE_LAST = RESET_TYPE_ROLLBACK,
} shepd_reset_type_t;
"""
class shepd_reset_type_t(enum.IntEnum):
	RESET_TYPE_HARD = 0
	RESET_TYPE_SOFT = 1
	RESET_TYPE_ROLLBACK = 2
	RESET_TYPE_FIRST = RESET_TYPE_HARD
	RESTE_TYPE_LAST = RESET_TYPE_ROLLBACK

"""
typedef enum shepd_sampler_hardware_type_e {
	SAMPLER_HW_TYPE_ARM = 0, /*!< Perform sampling by arm */
	SAMPLER_HW_TYPE_NVP,     /*!< Perform sampling by nvp */
	SAMPLER_HW_TYPE_NONE,    /*!< Don't perform sampling */
	SAMPLER_TYPE_FIRST = SAMPLER_HW_TYPE_ARM,
	SAMPLER_TYPE_LAST = SAMPLER_HW_TYPE_NONE,
} shepd_sample_hw_t;
"""
class shepd_sample_hw_t(enum.IntEnum):
	SAMPLER_HW_TYPE_ARM = 0
	SAMPLER_HW_TYPE_NVP = 1
	SAMPLER_HW_TYPE_NONE = 2
	SAMPLER_TYPE_FIRST = SAMPLER_HW_TYPE_ARM
	SAMPLER_TYPE_LAST = SAMPLER_HW_TYPE_NONE

"""
struct shepd_init_cfg {
	IN int log_level;          /*!< Log level */
	IN uint32_t reserved[63];  /*!< Reserved field */
};
"""
class shepd_init_cfg(ctypes.Structure):
	_fields_ = [
		("log_level", ctypes.c_int),
		("reserved", ctypes.c_uint32 * 63),
	]

""""
struct shepd_mem {
	IN void *virt;         /*!< Virtual address */
	IN unsigned long phys; /*!< Physical address */
	IN unsigned long size; /*!< Memory size */
};
"""
class shepd_mem(ctypes.Structure):
	_fields_ = [
		("virt", ctypes.c_void_p),
		("phys", ctypes.c_ulong),
		("size", ctypes.c_ulong),
	]
""""
struct shepd_io_dim {
	unsigned long plane;  /*!< The plane of the port */
	unsigned long depth;  /*!< The depth of the port */
	unsigned long height;  /*!< The height of the port */
	unsigned long width;  /*!< The width of the port */

	unsigned long pitch;  /*!< The pitch of the port in width alignment */
	uint32_t bitvector : 1;  /*!< If this port is a bit vector, one element is 1 bit in the DRAM if true */
	uint32_t is_variable : 1;  /*!< if it is scalar variable */
	uint32_t is_loop_pair : 1;  /*!< Indicate port is loop pair like LSTM layer. N = loop_cnt.
		* port_dram_size will be (N+1)*port_size, so there is one more port_size between batch port */
	uint32_t reserved_0 : 5;  /*!< Reserved field */

	uint32_t dram_fmt : 8;  /*!< The DRAM format of the port.  For internal use only */
	uint32_t loop_cnt : 16;  /*!< Indicator this port has loop cnt. N = loop_cnt.
		* port_dram_size will be N*port_size */

	uint32_t reserved_2[10];  /*!< Reserved field */
};
"""
class shepd_io_dim(ctypes.Structure):
	_fields_ = [
		("plane", ctypes.c_long),
		("depth", ctypes.c_long),
		("height", ctypes.c_long),
		("width", ctypes.c_long),
		("pitch", ctypes.c_long),
		("bitvector", ctypes.c_uint32, 1),
		("is_variable", ctypes.c_uint32, 1),
		("is_loop_pair", ctypes.c_uint32, 1),
		("reserved_0", ctypes.c_uint32, 5),
		("dram_fmt", ctypes.c_uint32, 8),
		("loop_cnt", ctypes.c_uint32, 16),
		("reserved_2", ctypes.c_uint32 * 10),
	]

"""
struct shepd_io_data_fmt {
	uint8_t sign;  /*!< 0: unsigned; 1: sign */
	uint8_t size;  /*!< 0: 8-bit;  1: 16-bit;  2: 32-bit;  3: 64-bit */
	int8_t expoffset;  /*!< Q value for quantized data */
	uint8_t expbits;  /*!< Exp bits for floating point, float element when it is non-zero */
	uint32_t bitsize; /*!< Bit size for one element, usually for special packed bit size.
			* 0: should refer to size; 1: bitvector; 4: 4bit; 8: 8bit(1byte);
			* 10: 10bit; 12: 12bit; 16: 16bit(2byte); 32: 32bit(4byte) */
};
"""
class shepd_io_data_fmt(ctypes.Structure):
	_fields_ = [
		("sign", ctypes.c_uint8),
		("size", ctypes.c_uint8),
		("expoffset", ctypes.c_uint8),
		("expbits", ctypes.c_uint8),
		("bitsize", ctypes.c_uint32),
	]
"""
struct shepd_img {
	OUT unsigned long size;          /*!< Image size */
	OUT struct shepd_io_dim dim;          /*!< The dimension information for the port */
	OUT struct shepd_io_data_fmt data_fmt;/*!< The data format of the port */
	IN struct shepd_mem img_mem;     /*!< Image memory description */
	IN uint32_t internal_cavalry_mem : 1; /*!< 0: The app allocates the cavalry memory and
		* specifies the physical address and size in img_mem. 1: Shepherd allocate
		* the cavalry memory, the app specifies virtual address and size in img_mem */
	IN uint32_t reserved_0 : 31;     /*!< Reserved field */
	IN uint32_t img_num;             /*!< Image num in img_mem */
	IN uint32_t reserved[31];        /*!< Reserved field */
};
"""
class shepd_img(ctypes.Structure):
	_fields_ = [
		("size", ctypes.c_ulong),
		("dim", shepd_io_dim),
		("data_fmt", shepd_io_data_fmt),
		("img_mem", shepd_mem),
		("internal_cavalry_mem", ctypes.c_uint32, 1),
		("reserved_0", ctypes.c_uint32, 31),
		("img_num", ctypes.c_uint32),
		("reserved", ctypes.c_uint32 * 31),
	]

"""
struct llava_extra {
	INOUT struct shepd_img vit_in; /*!< Vision transformer net input image information */
	IN char *vit_net_fn;           /*!< Vision transformer net filename */
	IN uint32_t patch_size;        /*!< Patch size */
	IN uint32_t img_start_token_id;/*!< Image start token */
	IN uint32_t img_end_token_id;  /*!< Image end token */
	IN uint32_t reserved[187];     /*!< Reserved field */
};
"""
class llava_extra(ctypes.Structure):
	_fields_ = [
		("vit_in", shepd_img),
		("vit_net_fn", ctypes.c_char_p),
		("patch_size", ctypes.c_uint32),
		("img_start_token_id", ctypes.c_uint32),
		("img_end_token_id", ctypes.c_uint32),
		("reserved", ctypes.c_uint32 * 187),
]

"""
struct llava_onevision_vit {
	IN int vit_mode;               /*!< 0: single image, 1: multi image, 2:video */
	IN uint32_t max_img_num;       /*!< Max image number for multi image and video mode */
	IN char *vit_net_fn;           /*!< Vit net filename */
	INOUT struct shepd_img vit_in; /*!< Vit input information */
	IN uint32_t reserved[4];              /*!< Reserved field */
};
"""
class llava_onevision_vit(ctypes.Structure):
	_fields_ = [
		("vit_mode", ctypes.c_int),
		("max_img_num", ctypes.c_uint32),
		("vit_net_fn", ctypes.c_char_p),
		("vit_in", shepd_img),
		("reserved", ctypes.c_uint32 * 4),
]

"""
struct llava_onevision_extra {
	INOUT struct llava_onevision_vit *vit; /*!< Vit array */
	IN uint32_t size;                      /*!< Size of vit array*/
	IN uint32_t index;                 /*!< Index of vit to run in vit array */
	IN uint32_t reserved[4];               /*!< Reserved field */
};
"""
class llava_onevision_extra(ctypes.Structure):
	_fields_ = [
		("vit", ctypes.c_void_p),         ## fixme: llava_onevision_vit * -> void *
		("size", ctypes.c_uint32),
		("index", ctypes.c_uint32),
		("reserved", ctypes.c_uint32 * 4),
]


"""
struct lisa_extra {
	INOUT struct shepd_img vit_in;  /*!< Vision transformer net input image information */
	INOUT struct shepd_img sam_enc; /*!< Segmentation encoder net input image information */
	OUT struct shepd_img sam_dec;   /*!< Segmentation decoder net output image information */
	IN char *vit_net_fn;            /*!< Vision transformer net filename */
	IN char *sam_enc_net_fn;        /*!< Segmentation encoder net filename */
	IN char *sam_dec_net_fn;        /*!< Segmentation decoder net filename */
	IN uint32_t patch_size;         /*!< Patch size */
	IN uint32_t img_start_token_id; /*!< Image start token */
	IN uint32_t img_end_token_id;   /*!< Image end token */
	IN uint32_t sam_dec_token_id;   /*!< Segmentation decoder token */
	IN uint32_t reserved[54];      /*!< Reserved field */
};
"""
class lisa_extra(ctypes.Structure):
	_fields_ = [
		("vit_in", shepd_img),
		("sam_enc", shepd_img),
		("sam_dec", shepd_img),
		("vit_net_fn", ctypes.c_char_p),
		("sam_enc_net_fn", ctypes.c_char_p),
		("sam_dec_net_fn", ctypes.c_char_p),
		("patch_size", ctypes.c_uint32),
		("img_start_token_id", ctypes.c_uint32),
		("img_end_token_id", ctypes.c_uint32),
		("sam_dec_token_id", ctypes.c_uint32),
		("reserved", ctypes.c_uint32 * 54),
	]

"""
union shepd_extra {
	struct llava_extra llava_ex; /*!< Llava extra configuration */
	struct lisa_extra lisa_ex;   /*!< Lisa extra configuration */
	struct llava_onevision_extra llava_onevision_ex; /*!< Llava onevision extra configuration */
};
"""
class shepd_extra(ctypes.Union):
	_fields_ = [
		("llava_ex", llava_extra),
		("lisa_ex", lisa_extra),
		("llava_onevision_ex", llava_onevision_extra),
	]

"""
struct shepd_config {
	IN const char *model_path;      /*!< Model path */
	IN uint32_t batch_size;         /*!< Batch size */
	IN uint32_t max_user_num;       /*!< Max user num */
	IN struct shepd_device_cfg device;  /*!< The device configuration */
	INOUT union shepd_extra shepd_ex;   /*!< The model extra descriptor */
	OUT uint32_t max_seq_length;        /*!< Model parameters: max sequence length */
	OUT uint32_t vocab_size;            /*!< Model parameters: vocabulary size */
	OUT uint32_t eos_token_id;          /*!< Model parameters: end-of-sequence token id */
	IN uint32_t reserved[505];          /*!< Reserved field */
};
"""
class shepd_config(ctypes.Structure):
	_fields_ = [
		("model_path", ctypes.c_char_p),
		("batch_size", ctypes.c_uint32),
		("max_user_num", ctypes.c_uint32),
		("device", shepd_device_cfg),
		("shepd_ex", shepd_extra),
		("max_seq_length", ctypes.c_uint32),
		("vocab_size", ctypes.c_uint32),
		("eos_token_id", ctypes.c_uint32),
		("reserved", ctypes.c_uint32 * 505),
	]

"""
struct tokenizer_enc_cfg {
	IN uint32_t no_sys_prompt : 1; /*!< Don't apply system prompt on input text */
	IN uint32_t reserved_0 : 31;   /*!< Reserved field */
	IN uint32_t reserved[31];      /*!< Reserved field */
};
"""
class tokenizer_enc_cfg(ctypes.Structure):
	_fields_ = [
		("no_sys_prompt", ctypes.c_uint32, 1),
		("reserved_0", ctypes.c_uint32, 31),
		("reserved", ctypes.c_uint32 * 31),
	]

"""
struct tokenizer_dec_cfg {
	IN uint32_t no_piece_mode : 1; /*!< Decoding all token ids one-time, instead of decoding token one by one */
	IN uint32_t reserved_0 : 31;   /*!< Reserved field */
	IN uint32_t reserved[32];            /*!< Reserved field */
};
"""
class tokenizer_dec_cfg(ctypes.Structure):
	_fields_ = [
		("no_piece_mode", ctypes.c_uint32, 1),
		("reserved_0", ctypes.c_uint32, 31),
		("reserved", ctypes.c_uint32 * 32),
	]

"""
struct tokenizer_dec_res {
	OUT const char *text;  /*!< Decoded text */
	OUT unsigned long len; /*!< Length of decoded text, null character not included*/
};
"""
class tokenizer_dec_res(ctypes.Structure):
	_fields_ = [
		("text", ctypes.c_char_p),
		("len", ctypes.c_ulong),
	]

"""
struct token_id_list {
	INOUT uint32_t *ids;       /*!< Token ids */
	INOUT uint32_t num;        /*!< Token ids num */
};
"""
class token_id_list(ctypes.Structure):
	_fields_ = [
		("ids", ctypes.c_void_p), ##fixme: uint32_t * -> void *
		("num", ctypes.c_uint32),
	]

"""
struct preproc_res {
	OUT struct token_id_list id_list;  /*!< Token id list */
	OUT uint32_t reserved[29];         /*!< Reserved field */
};
"""
class preproc_res(ctypes.Structure):
	_fields_ = [
		("id_list", token_id_list),
		("reserved", ctypes.c_uint32 * 29),
	]

"""
struct shepd_run_cfg {
	IN float top_p;                   /*!< Top-P */
	IN float temperature;             /*!< Temperature */
	shepd_sample_hw_t sample_hw_type; /*!< Sampler hardware type */
	IN uint32_t query_logits_en : 1;  /*!< Flag to get logits information */
	IN uint32_t single_turn_conversation : 1;/*!< Single turn conversation, for performance test only */
	IN uint32_t reserved_0 : 30;      /*!< Reserved field */
	IN uint32_t force_token_id;       /*!< Input token id, typically for external sample case */
	IN uint8_t priority;              /*!< Priority of task, lowest(0)->highest(31) */
	IN uint8_t reserved_2[3];         /*!< Reserved field */
	IN uint32_t reserved[58];         /*!< Reserved field */
};
"""
class shepd_run_cfg(ctypes.Structure):
	_fields_ = [
		("top_p", ctypes.c_float),
		("temperature", ctypes.c_float),
		("sample_hw_type", ctypes.c_int),
		("query_logits_en", ctypes.c_uint32, 1),
		("single_turn_conversation", ctypes.c_uint32, 1),
		("reserved_0", ctypes.c_uint32, 30),
		("force_token_id", ctypes.c_uint32),
		("priority", ctypes.c_uint8),
		("reserved_2", ctypes.c_uint8 * 3),
		("reserved", ctypes.c_uint32 * 58),
	]

"""
struct shepd_output {
	OUT uint32_t token_id;     /*!< Output token */
	OUT uint32_t pos;          /*!< Position in inference context window */
	OUT const void *logits_virt;  /*!< Logits memory virtual address, return NULL if query_logits_en = 0 */
	OUT uint32_t logits_mem_size; /*!< Logits memory size, return 0 if query_logits_en = 0 */
	OUT uint32_t logits_elem_size;/*!< Logits element size, return 0 if query_logits_en = 0 */
	OUT float prefill_time;       /*!< Prefill time, uint: s */
	OUT float output_time;        /*!< Output one token time, uint: s */
	OUT uint32_t reserved[120];   /*!< Reserved field */
};
"""
class shepd_output(ctypes.Structure):
	_fields_ = [
		("token_id", ctypes.c_uint32),
		("pos", ctypes.c_uint32),
		("logits_virt", ctypes.c_void_p),
		("logits_mem_size", ctypes.c_uint32),
		("logits_elem_size", ctypes.c_uint32),
		("prefill_time", ctypes.c_float),
		("output_time", ctypes.c_float),
		("reserved", ctypes.c_uint32 * 120),
	]

"""
struct shepd_reset_cfg {
	IN shepd_reset_type_t reset_type; /*!< Reset type */
	OUT uint32_t reset_pos;   /*!< Reset position in inference context window */
	IN uint32_t roll_back_pos;/*!< Rollback postion in inference context window */
	IN uint32_t reserved[61]; /*!< Reserved field */
};
"""
class shepd_reset_cfg(ctypes.Structure):
	_fields_ = [
		("reset_type", ctypes.c_int),
		("reset_pos", ctypes.c_uint32),
		("roll_back_pos", ctypes.c_uint32),
		("reserved", ctypes.c_uint32 * 61),
	]

""""
typedef enum dump_type_e {
	DUMP_ALL_MODE = 0,       /*!< Dump all layers data and logits*/
	DUMP_LAYER_MODE,         /*!< Dump a layer's attention and mlp data*/
	DUMP_LOGITS_MODE,        /*!< Dump logits */
	DUMP_TYPE_FIRST = DUMP_LAYER_MODE,
	DUMP_TYPE_LAST = DUMP_LOGITS_MODE,
} dump_type_t;
"""
class dump_type_t(enum.IntEnum):
	DUMP_ALL_MODE = 0
	DUMP_LAYER_MODE = 1
	DUMP_LOGITS_MODE = 2
	DUMP_TYPE_FIRST = DUMP_LAYER_MODE
	DUMP_TYPE_LAST = DUMP_LOGITS_MODE

""""
typedef enum perf_type_e {
	PERF_LAYER_MODE = 0, /*!< Print performace by layer */
	PERF_TOKEN_MODE,     /*!< Print performace by token */
	PERF_TYPE_FIRST = PERF_LAYER_MODE,
	PERF_TYPE_LAST = PERF_TOKEN_MODE,
} perf_type_t;
"""
class perf_type_t(enum.IntEnum):
	PERF_LAYER_MODE = 0
	PERF_TOKEN_MODE = 1
	PERF_TYPE_FIRST = PERF_LAYER_MODE
	PERF_TYPE_LAST = PERF_TOKEN_MODE

""""
struct shepd_dump_cfg {
	dump_type_t dump_type; /*!< Dump type */
	uint32_t layer_id;     /*!< Layer index, only valid when dump_type equals DUMP_LAYER_MODE */
	uint32_t reserved[62]; /*!< Reserved field */
};
"""
class shepd_dump_cfg(ctypes.Structure):
	_fields_ = [
		("dump_type", ctypes.c_int),
		("layer_id", ctypes.c_uint32),
		("reserved", ctypes.c_uint32 * 62),
	]

"""
struct shepd_perf_cfg {
	perf_type_t perf_type; /*!< Performance print type */
	uint32_t layer_id;     /*!< Layer index */
	uint32_t reserved[62]; /*!< Reserved field */
};
"""
class shepd_perf_cfg(ctypes.Structure):
	_fields_ = [
		("perf_type", ctypes.c_int),
		("layer_id", ctypes.c_uint32),
		("reserved", ctypes.c_uint32 * 62),
	]

"""
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
	LOG_INFO =2
	LOG_DEBUG = 3
	LOG_VERBOSE = 4
	LOG_DEFAULT = LOG_INFO

"""
struct shepherd_version {
	uint32_t major;       /*!< Version major number */
	uint32_t minor;       /*!< Version minor number */
	uint32_t patch;       /*!< Version patch number */
	uint32_t mod_time;    /*!< Version modification time */
	char description[64]; /*!< Version description */
};
"""
class shepherd_version(ctypes.Structure):
	_fields_ = [
		("major", ctypes.c_uint32),
		("minor", ctypes.c_uint32),
		("patch", ctypes.c_uint32),
		("mod_time", ctypes.c_uint32),
		("description", ctypes.c_char * 64),
	]
