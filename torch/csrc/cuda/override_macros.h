#include "undef_macros.h"

#define THStoragePtr THCStoragePtr
#define THPStoragePtr THCPStoragePtr
#define THTensorPtr THCTensorPtr
#define THPTensorPtr THCPTensorPtr

#define THStorage THCStorage
#define THStorage_(NAME) THCStorage_(NAME)
#define THTensor THCTensor
#define THTensor_(NAME) THCTensor_(NAME)

#define THPStorage_(NAME) TH_CONCAT_4(THCP,Real,Storage_,NAME)
#define THPStorage THCPStorage
#define THPStorageBaseStr THCPStorageBaseStr
#define THPStorageStr THCPStorageStr
#define THPStorageClass THCPStorageClass
#define THPStorageType THCPStorageType

#define THPTensor_(NAME) TH_CONCAT_4(THCP,Real,Tensor_,NAME)
#define THPTensor_stateless_(NAME) TH_CONCAT_4(THCP,Real,Tensor_stateless_,NAME)
#define THPTensor THCPTensor
#define THPTensorStr THCPTensorStr
#define THPTensorBaseStr THCPTensorBaseStr
#define THPTensorClass THCPTensorClass
#define THPTensorType THCPTensorType

#define THPTensorStatelessType THCPTensorStatelessType
#define THPTensorStateless THCPTensorStateless

#define LIBRARY_STATE_NOARGS state
#define LIBRARY_STATE state,
#define TH_GENERIC_FILE THC_GENERIC_FILE

#define THHostTensor TH_CONCAT_3(TH,Real,Tensor)
#define THHostTensor_(NAME) TH_CONCAT_4(TH,Real,Tensor_,NAME)
#define THHostStorage TH_CONCAT_3(TH,Real,Storage)
#define THHostStorage_(NAME) TH_CONCAT_4(TH,Real,Storage_,NAME)
