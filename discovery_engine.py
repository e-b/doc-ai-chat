import config as cfg
from google.cloud import aiplatform
from google.cloud import discoveryengine_v1alpha as discoveryengine
from google.api_core.client_options import ClientOptions
from google.cloud import storage
from google.cloud.storage.blob import Blob


storage_client = storage.Client()

aiplatform.init(
    project=cfg.PROJECT_ID, location=cfg.REGION, staging_bucket=cfg.BUCKET_URI
)


location = "eu"
client_options = (ClientOptions(api_endpoint=f"{location}-discoveryengine.googleapis.com") if location != 'global' else None)
summary_search_client = discoveryengine.SearchServiceClient(client_options=client_options)
summary_serving_config = summary_search_client.serving_config_path(project=cfg.PROJECT_ID,
                                                                   location=location, # or global
                                                                   data_store="impp-med-docs_1720175654352", #"impp-med-docs_1720175654352",
                                                                   serving_config="default_search")

summary_spec = discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec(
        # Use top 5 results for summary
        summary_result_count=5,
        include_citations=True)

content_search_spec = discoveryengine.SearchRequest.ContentSearchSpec(
    snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(return_snippet=True),
    summary_spec=summary_spec,
    extractive_content_spec=discoveryengine.SearchRequest.ContentSearchSpec.ExtractiveContentSpec(
        max_extractive_answer_count=5,   # turns on extractive content
        max_extractive_segment_count=5,  # extractive segments
        return_extractive_segment_score=True # private preview
    )
)

def get_doc_url(uri):
    """Transform the uri of the document into an url that can be accessed via browser

    Args:
        uri (str): uri of the document

    Returns:
        Browser accessible url of the document
    """
    blob = Blob.from_string(uri, storage_client)
    url = blob.public_url
    return url.replace('googleapis', 'mtls.cloud.google')

def search(query, filter: list[dict] = []):
    """Executes the summary search algorithm by calling Vertex AI Search with summarization. Returns a summary of the result plus a list of documents, snippets, extracts and segments.
    
    Args:
        query (str): Query from user
        filter (list[dict]): Filters to use

    Returns:
        dict: {'response': (str) Answer text, 'documents': (list[dict]) List of documents}
    """
    filter_str = make_filter_str(filter)
    request = discoveryengine.SearchRequest(
        serving_config=summary_serving_config,
        query=query,
        filter=filter_str,
        content_search_spec=content_search_spec,
        query_expansion_spec=discoveryengine.SearchRequest.QueryExpansionSpec(
            condition=discoveryengine.SearchRequest.QueryExpansionSpec.Condition.AUTO
        ),
        spell_correction_spec=discoveryengine.SearchRequest.SpellCorrectionSpec(
            mode=discoveryengine.SearchRequest.SpellCorrectionSpec.Mode.SUGGESTION_ONLY
        )
    )
    search_response = summary_search_client.search(request)
    response = {'response': search_response.summary.summary_text, 'documents': []}
    for i, result in enumerate(search_response.results, 1):
        struct_data = result.document.derived_struct_data
        doc = {'name': f'[{i}] ' + struct_data['link'], 'url': get_doc_url(struct_data['link']), 'snippets': [], 'extracts': [], 'segments': []}
        for snippet in struct_data.get('snippets', []):
            doc['snippets'].append(snippet['snippet'])
        for extract in struct_data.get('extractive_answers', []):
            doc['extracts'].append({'pageNumber': extract.get('pageNumber'), 'extract': extract.get('content')})
        for extract in struct_data.get('extractive_segments', []):
            doc['segments'].append({'pageNumber': extract.get('pageNumber'), 'extract': extract.get('content'), 'relevanceScore': extract.get('relevanceScore')})
        response['documents'].append(doc)
    return response


def make_filter_str(filter: list[dict]):
    """ Create a filter string to be used in a search request
    
    Args:
        filter (list[dict]): List of filters

    Returns:
        str: Filter string for a search request
    """
    filter_str = []
    for f in filter:
        values = ', '.join([f'"{x}"' for x in list(f.values())[0]])
        filter_str.append(f'{list(f.keys())[0]}: ANY({values})')
    return " AND ".join(filter_str)