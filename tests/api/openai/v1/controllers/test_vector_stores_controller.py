from httpx import AsyncClient

async def test_cancel_vector_store_file_batch(client: AsyncClient):
    response = await client.post(
        "/api/openai/v1/vector_stores/vector_store_id_example/file_batches/batch_id_example/cancel",
    )

    assert response.status_code == 401

async def test_create_vector_store(client: AsyncClient):
    response = await client.post(
        "/api/openai/v1/vector_stores",
    )

    assert response.status_code == 401

async def test_create_vector_store_file(client: AsyncClient):
    response = await client.post(
        "/api/openai/v1/vector_stores/vs_abc123/files",
    )

    assert response.status_code == 401

async def test_create_vector_store_file_batch(client: AsyncClient):
    response = await client.post(
        "/api/openai/v1/vector_stores/vs_abc123/file_batches",
    )

    assert response.status_code == 401

async def test_delete_vector_store(client: AsyncClient):
    response = await client.delete(
        "/api/openai/v1/vector_stores/vector_store_id_example",
    )

    assert response.status_code == 401

async def test_delete_vector_store_file(client: AsyncClient):
    response = await client.delete(
        "/api/openai/v1/vector_stores/vector_store_id_example/files/file_id_example",
    )

    assert response.status_code == 401

async def test_get_vector_store(client: AsyncClient):
    response = await client.get(
        "/api/openai/v1/vector_stores/vector_store_id_example",
    )

    assert response.status_code == 401

async def test_get_vector_store_file(client: AsyncClient):
    response = await client.get(
        "/api/openai/v1/vector_stores/vs_abc123/files/file-abc123",
    )

    assert response.status_code == 401

async def test_get_vector_store_file_batch(client: AsyncClient):
    response = await client.get(
        "/api/openai/v1/vector_stores/vs_abc123/file_batches/vsfb_abc123",
    )

    assert response.status_code == 401

async def test_list_files_in_vector_store_batch(client: AsyncClient):
    response = await client.get(
        "/api/openai/v1/vector_stores/vector_store_id_example/file_batches/batch_id_example/files?limit=20&order=desc&after=after_example&before=before_example&filter=filter_example",
    )

    assert response.status_code == 401

async def test_list_vector_store_files(client: AsyncClient):
    response = await client.get(
        "/api/openai/v1/vector_stores/vector_store_id_example/files?limit=20&order=desc&after=after_example&before=before_example&filter=filter_example",
    )

    assert response.status_code == 401

async def test_list_vector_stores(client: AsyncClient):
    response = await client.get(
        "/api/openai/v1/vector_stores?limit=20&order=desc&after=after_example&before=before_example",
    )

    assert response.status_code == 401

async def test_modify_vector_store(client: AsyncClient):
    response = await client.post(
        "/api/openai/v1/vector_stores/vector_store_id_example",
    )

    assert response.status_code == 401
