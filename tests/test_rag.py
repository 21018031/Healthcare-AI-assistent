from pathlib import Path

from app.rag import chunk_documents, load_documents, read_medlineplus_xml


def test_load_documents_reads_text_files(tmp_path: Path) -> None:
    document_path = tmp_path / "policy.txt"
    document_path.write_text("Report incidents within one business day.", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert len(documents) == 1
    assert documents[0].source == str(document_path)
    assert "business day" in documents[0].text


def test_chunk_documents_adds_source_and_stable_ids(tmp_path: Path) -> None:
    document_path = tmp_path / "policy.txt"
    document_path.write_text("A " * 100, encoding="utf-8")
    documents = load_documents(tmp_path)

    chunks = chunk_documents(documents, chunk_size=50, chunk_overlap=10)

    assert len(chunks) > 1
    assert all(chunk.source == str(document_path) for chunk in chunks)
    assert len({chunk.id for chunk in chunks}) == len(chunks)


def test_read_medlineplus_xml_creates_topic_documents(tmp_path: Path) -> None:
    xml_path = tmp_path / "mplus_topics.xml"
    xml_path.write_text(
        """
        <health-topics>
          <health-topic title="Asthma" url="https://medlineplus.gov/asthma.html">
            <also-called>Bronchial asthma</also-called>
            <full-summary>Asthma is a chronic disease that affects your airways.</full-summary>
          </health-topic>
        </health-topics>
        """,
        encoding="utf-8",
    )

    documents = read_medlineplus_xml(xml_path)

    assert len(documents) == 1
    assert "Asthma" in documents[0].source
    assert "medlineplus.gov/asthma.html" in documents[0].source
    assert "chronic disease" in documents[0].text
