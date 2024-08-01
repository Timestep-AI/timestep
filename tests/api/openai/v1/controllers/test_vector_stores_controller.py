from fastapi.testclient import TestClient

from timestep.server import fastapi_app

client = TestClient(fastapi_app)

def test_cancel_vector_store_file_batch():
    response = client.post(
        "/api/openai/v1/vector_stores/vector_store_id_example/file_batches/batch_id_example/cancel",
    )

    assert response.status_code == 501

def test_create_vector_store():
    response = client.post(
        "/api/openai/v1/vector_stores",
    )

    assert response.status_code == 501

def test_create_vector_store_file():
    response = client.post(
        "/api/openai/v1/vector_stores/vs_abc123/files",
    )

    assert response.status_code == 501

def test_create_vector_store_file_batch():
    response = client.post(
        "/api/openai/v1/vector_stores/vs_abc123/file_batches",
    )

    assert response.status_code == 501

def test_delete_vector_store():
    response = client.delete(
        "/api/openai/v1/vector_stores/vector_store_id_example",
    )

    assert response.status_code == 501

def test_delete_vector_store_file():
    response = client.delete(
        "/api/openai/v1/vector_stores/vector_store_id_example/files/file_id_example",
    )

    assert response.status_code == 501

def test_get_vector_store():
    response = client.get(
        "/api/openai/v1/vector_stores/vector_store_id_example",
    )

    assert response.status_code == 501

def test_get_vector_store_file():
    response = client.get(
        "/api/openai/v1/vector_stores/vs_abc123/files/file-abc123",
    )

    assert response.status_code == 501

def test_get_vector_store_file_batch():
    response = client.get(
        "/api/openai/v1/vector_stores/vs_abc123/file_batches/vsfb_abc123",
    )

    assert response.status_code == 501

def test_list_files_in_vector_store_batch():
    response = client.get(
        "/api/openai/v1/vector_stores/vector_store_id_example/file_batches/batch_id_example/files?limit=20&order=desc&after=after_example&before=before_example&filter=filter_example",
    )

    assert response.status_code == 501

def test_list_vector_store_files():
    response = client.get(
        "/api/openai/v1/vector_stores/vector_store_id_example/files?limit=20&order=desc&after=after_example&before=before_example&filter=filter_example",
    )

    assert response.status_code == 501

def test_list_vector_stores():
    response = client.get(
        "/api/openai/v1/vector_stores?limit=20&order=desc&after=after_example&before=before_example",
    )

    assert response.status_code == 501

def test_modify_vector_store():
    response = client.post(
        "/api/openai/v1/vector_stores/vector_store_id_example",
    )

    assert response.status_code == 501
