from io import BytesIO
import nft_storage
from nft_storage.api import nft_storage_api
from pprint import pprint
from dotenv import load_dotenv
import os

load_dotenv()
configuration = nft_storage.Configuration(
    access_token = os.environ.get('nftStorageAccessToken')
)


def store_NFTStorage(fileData: BytesIO) -> str:
    with nft_storage.ApiClient(configuration) as api_client:
        api_instance = nft_storage_api.NFTStorageAPI(api_client)
        try:
            api_response = api_instance.store(
                fileData, _check_return_type=False)
            pprint(api_response)
        except nft_storage.ApiException as e:
            print("Exception when calling NFTStorageAPI->store: %s\n" % e)

    return api_response.value['cid']
