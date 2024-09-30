#define size_t int
struct ibv_pd *pd;
struct ibv_mr *mr;



// Memory Reg Example

struct ibv_mr *ibv_reg_mr(struct ibv_pd *pd,
					    void *addr, size_t length,
					    enum ibv_access_flags access);

/* Notice the following fields in struct ibv_mr:
rkey  - The remote key of this MR
lkey  - The local key of this MR
addr - The start address of the memory buffer that this MR registered
length - The size of the memory buffer that was registered
*/

//Deregister a Memory Region

int ibv_dereg_mr(struct ibv_mr *mr);




/*
This verb should be called if there is no outstanding
Send Request or Receive Request that points to it
*/

#define size_t int
#define uint64_t int
#define uint32_t int
struct ibv_pd *pd;
struct ibv_mr *mr;

//Scatter Gather Entry

struct ibv_sge {
	uint64_t addr; // Start address of the memory buffer (registered memory)
	uint32_t length; // Size (in bytes) of the memory buffer
	uint32_t lkey;   // lkey of Memory Region associated with this memory buffer
};












//Post Send

int ibv_post_send(struct ibv_qp *qp, struct ibv_send_wr *wr,
                  struct ibv_send_wr **bad_wr);
struct ibv_send_wr {
	uint64_t wr_id;            // Private context that will be available in the corresponding Work Completion
	struct ibv_send_wr *next;  // Address of the next Send Request. NULL in the last Send Request
	struct ibv_sge *sg_list;   // Array of scatter/gather elements
	int num_sge;               // Number of elements in sg_list
	enum ibv_wr_opcode opcode; // The opcode to be used
	int send_flags;            // Send flags. Or of the following flags:

    /* IBV_SEND_FENCE - Prevent process this Send Request until the processing of previous RDMA
    //                   Read and Atomic operations were completed.
    //IBV_SEND_SIGNALED - Generate a Work Completion after processing of this Send Request ends
    //IBV_SEND_SOLICITED - Generate Solicited event for this message in remote side
    IBV_SEND_INLINE  - allow the low-level driver to read the gather buffers*/

	uint32_t imm_data;  // Send message with immediate data (for supported opcodes)
union {
	 struct {                           // Attributes for RDMA Read and write opcodes
		uint64_t remote_addr;      // Remote start address (the message size is according to the S/G entries)
		uint32_t rkey;             // rkey of Memory Region that is associated with remote memory buffer
	 } rdma;
	 struct {                           // Attributes for Atomic opcodes
		uint64_t remote_addr;      // Remote start address (the message size is according to the S/G entries)
		uint64_t compare_add;      // Value to compare/add (depends on opcode)
		uint64_t swap;             // Value to swap if the comparison passed
		uint32_t rkey;             // rkey of Memory Region that is associated with remote memory buffer
	} atomic;
}
};



//Post Receive Request

int ibv_post_recv(struct ibv_qp *qp, struct ibv_recv_wr *wr,
                  struct ibv_recv_wr **bad_wr);

//Post Send Request

int ibv_post_send(struct ibv_qp *qp, struct ibv_send_wr *wr,
                  struct ibv_send_wr **bad_wr);

/* Warning: bad_wr is mandatory; It will be assigned with the address of
the Receive Request that its posting failed */

struct ibv_recv_wr {
	uint64_t wr_id;           // Private context, available in the corresponding Work Completion
	struct ibv_recv_wr *next; // Address of the next Receive Request. NULL in the last Request
	struct ibv_sge *sg_list;  // Array of scatter elements
	int num_sge;              // Number of elements in sg_list
};



// Polling for work completion

int ibv_poll_cq(struct ibv_cq *cq, int num_entries, struct ibv_wc *wc);

//Work Completion for each entry

struct ibv_wc {
	uint64_t wr_id;                // Private context that was posted in the corresponding Work Request
	enum ibv_wc_status status;     // The status of the Work Completion
	enum ibv_wc_opcode opcode;     // The opcode of the Work Completion
	uint32_t vendor_err;           // Vendor specific error syndrome
	uint32_t byte_len;             // Number of bytes that were received
	uint32_t imm_data;             // Immediate data, in network order, if the flags indicate that such exists
	uint32_t qp_num;               // The local QP number that this Work Completion ended in
	uint32_t src_qp;               // The remote QP number
	int wc_flags;                  // Work Completion flags. Or of the following flags:

     /* IBV_WC_GRH ג€“ Indicator that the first 40 bytes of the receive buffer(s) contain a valid GRH
      IBV_WC_WITH_IMM ג€“ Indicator that the received message contains immediate data */

	uint16_t pkey_index;
	uint16_t slid;                                // For UD QP: the source LID
	uint8_t sl;                                     // For UD QP: the source Service Level
	uint8_t dlid_path_bits;                      // For UD QP: the destination LID path bits
};


// typical completion statuses

IBV_WC_SUCCESS ג€“ Operation completed successfully
IBV_WC_LOC_LEN_ERR ג€“ Local length error when processing SR or RR
IBV_WC_LOC_PROT_ERR ג€“ Local Protection error; S/G entries doesnג€™t point to a valid MR
IBV_WC_WR_FLUSH_ERR ג€“ Work Request flush error; it was processed when the QP was in Error state

IBV_WC_RETRY_EXC_ERR ג€“ Retry exceeded; the remote QP didnג€™t send any ACK/NACK, even after
            message retransmission

IBV_WC_RNR_RETRY_EXC_ERR ג€“ Receiver Not Ready; a message that requires a Receive Request
           was sent, but isnג€™t any RR in the remote QP, even after message retransmission
