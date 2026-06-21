from pathlib import Path


def test_read_tsv_with_fallback_uses_gbk_when_utf8_fails(tmp_path: Path):
    from scripts.dre_nips_readers.encoding import read_tsv_with_fallback

    path = tmp_path / "participants.tsv"
    path.write_bytes("participant_id\toutcome\t备注\nsub-P001\tS\t成功\n".encode("gbk"))

    df = read_tsv_with_fallback(path)

    assert df.loc[0, "participant_id"] == "sub-P001"
    assert df.loc[0, "备注"] == "成功"
