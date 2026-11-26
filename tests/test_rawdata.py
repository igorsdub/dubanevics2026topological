"""Raw-data acquisition: translation, transforms, checksums, zip extraction, checks.

Tiny synthetic fixtures only; nothing here touches the network or R.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import polars as pl
import pytest

from dubanevics2026topological.rawdata import (
    check_raw,
    fetch_data,
    files,
    make_raw,
    papkou_fetch,
)
from dubanevics2026topological.rawdata.value_digest import bytes_sha256, value_digest


def md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


# ---- translation --------------------------------------------------------------------


def test_translation_standard_code_and_stop_symbols():
    frame = pl.DataFrame({"d": ["ATGGCCAAA", "TAATAGTGA", "TGGTGTTTT", "GCCGATCTC"]})
    out = frame.select(aa=make_raw.translate_expr("d"))["aa"].to_list()
    assert out == ["MAK", "%^&", "WCF", "ADL"]


def test_translation_covers_all_64_codons():
    assert len(make_raw.CODON_TABLE) == 64
    assert sorted(set(make_raw.CODON_TABLE.values())) == sorted(
        set("ACDEFGHIKLMNPQRSTVWY%^&")
    )


def test_translation_rejects_unknown_codon():
    with pytest.raises(pl.exceptions.PolarsError):
        pl.DataFrame({"d": ["ATGNNNAAA"]}).select(make_raw.translate_expr("d"))


# ---- transforms ---------------------------------------------------------------------


def test_wt_drops_wt_row_and_keeps_order_and_dtypes():
    table = pl.DataFrame(
        {
            "SV": ["AAAAAAAAA", "GCCGATCTC", "TAATAGTGA"],
            "m": [-0.5, 0.0, 0.25],
            "m_se": [0.1, 0.2, 0.3],
            "m_p.value": [1e-9, 1.0, 0.5],
        }
    )
    out = make_raw.transform_wt(table)
    assert out.columns == list(make_raw.FITNESS_SCHEMA)
    assert dict(out.schema) == make_raw.FITNESS_SCHEMA
    assert out["dna_to"].to_list() == ["AAAAAAAAA", "TAATAGTGA"]
    assert set(out["dna_from"]) == {"GCCGATCTC"}
    assert out["aa_from"].to_list() == ["ADL", "ADL"]
    assert out["aa_to"].to_list() == ["KKK", "%^&"]
    assert out["s"].to_list() == [-0.5, 0.25]
    assert out["se"].to_list() == [0.1, 0.3]
    assert out["pval"].to_list() == [1e-9, 0.5]


def test_diff_swaps_negative_edges_and_takes_absolute_value():
    table = pl.DataFrame(
        {
            "from": ["AAAAAAAAA", "AAAAAAAAC", "AAAAAAAAG"],
            "to": ["AAAAAAAAC", "AAAAAAAAG", "AAAAAAAAT"],
            "m": [0.5, -0.25, 0.0],
            "m_se": [0.1, 0.2, 0.3],
            "m_p.value": [0.01, 0.02, 0.03],
        }
    )
    out = make_raw.transform_diff(table)
    assert dict(out.schema) == make_raw.FITNESS_SCHEMA
    assert out["dna_from"].to_list() == ["AAAAAAAAA", "AAAAAAAAG", "AAAAAAAAG"]
    assert out["dna_to"].to_list() == ["AAAAAAAAC", "AAAAAAAAC", "AAAAAAAAT"]
    assert out["s"].to_list() == [0.5, 0.25, 0.0]
    assert out["aa_from"].to_list() == ["KKK", "KKK", "KKK"]
    assert out["aa_to"].to_list() == ["KKN", "KKN", "KKN"]


def test_peaks_are_out_degree_zero_nodes_sorted():
    nodes = pl.DataFrame(
        {"name": ["CCC", "AAA", "GGG", "TTT"], "out_degree": [0, 2, 1, 0]}
    )
    assert make_raw.peaks_from_nodes(nodes)["dna"].to_list() == ["CCC", "TTT"]
    assert make_raw.peaks_from_nodes(nodes).columns == ["dna"]


def test_read_interim_roundtrips_exact_doubles(tmp_path: Path):
    value = 0.1 + 0.2
    path = tmp_path / "t.csv"
    path.write_text(f'"SV","m"\n"AAA","{value:.17g}"\n"CCC",\n')
    frame = make_raw.read_interim(path, ("SV",))
    assert frame["m"].to_list() == [value, None]


# ---- write_outputs, shape check, comparison -----------------------------------------

GRAPHML = """<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <key id="v_name" for="node" attr.name="name" attr.type="string"/>
  <key id="v_m" for="node" attr.name="m" attr.type="double"/>
  <key id="e_d" for="edge" attr.name="hamm.dist" attr.type="double"/>
  <graph id="G" edgedefault="directed">
    <node id="n0"><data key="v_name">AAAAAAAAA</data><data key="v_m">{m}</data></node>
    <node id="n1"><data key="v_name">AAAAAAAAC</data></node>
    <edge source="n0" target="n1"><data key="e_d">1</data></edge>
  </graph>
</graphml>
"""


def tiny_interim(tmp_path: Path, graph_m: str = "0.5") -> Path:
    d = tmp_path / "interim"
    d.mkdir()
    (d / "fitness_wt.csv").write_text(
        '"SV","m","m_se","m_p.value"\n"AAAAAAAAA","-0.5","0.1","0.001"\n"GCCGATCTC","0","0","1"\n'
    )
    (d / "fitness_diff.csv").write_text(
        '"from","to","m","m_se","m_p.value"\n'
        '"AAAAAAAAA","AAAAAAAAC","-0.25","0.1","0.01"\n"AAAAAAAAC","AAAAAAAAG","0.5","0.1","0.02"\n'
    )
    (d / "graph_nodes.csv").write_text(
        '"name","out_degree"\n"AAAAAAAAA",1\n"AAAAAAAAC",0\n'
    )
    (d / "graph.graphml").write_text(GRAPHML.format(m=graph_m))
    return d


EXPECTED = {
    "graph_nodes": 2,
    "graph_edges": 1,
    "peaks": 1,
    "wt_rows": 1,
    "diff_rows": 2,
}


def test_write_outputs_and_shape_check(tmp_path: Path):
    out = tmp_path / "raw"
    make_raw.write_outputs(tiny_interim(tmp_path), out)
    assert sorted(p.name for p in out.iterdir()) == sorted(make_raw.RAW_FILES)
    assert make_raw.check_shapes(out, EXPECTED) == []
    diff = pl.read_parquet(out / "papkou2023_fitness_diff.parquet")
    assert diff["dna_from"].to_list() == ["AAAAAAAAC", "AAAAAAAAC"]
    assert pl.read_csv(out / "papkou2023_peaks.csv")["dna"].to_list() == ["AAAAAAAAC"]


def test_shape_check_reports_every_mismatch(tmp_path: Path):
    out = tmp_path / "raw"
    make_raw.write_outputs(tiny_interim(tmp_path), out)
    problems = make_raw.check_shapes(
        out, {**EXPECTED, "wt_rows": 9, "peaks": 5, "graph_nodes": 7}
    )
    assert len(problems) == 3
    (out / "papkou2023_peaks.csv").unlink()
    assert any("missing" in p for p in make_raw.check_shapes(out, EXPECTED))


def test_check_against_equal_and_different(tmp_path: Path):
    a, b = tmp_path / "a", tmp_path / "b"
    make_raw.write_outputs(tiny_interim(tmp_path), a)
    (tmp_path / "x").mkdir()
    make_raw.write_outputs(tiny_interim(tmp_path / "x"), b)
    assert check_raw.check_against(a, b) == []
    # change a node attribute in the graph and a value in the table
    (b / "papkou2023_graph.graphml").write_text(GRAPHML.format(m="0.6"))
    frame = pl.read_parquet(b / "papkou2023_fitness_wt.parquet").with_columns(
        s=pl.col("s") + 1
    )
    frame.write_parquet(b / "papkou2023_fitness_wt.parquet")
    (b / "papkou2023_peaks.csv").write_text("dna\nAAAAAAAAA\n")
    problems = check_raw.check_against(a, b)
    assert len(problems) == 3
    assert any("fitness_wt" in p for p in problems)
    assert any("peaks" in p for p in problems)
    assert any("1 nodes differ" in p for p in problems)


def test_value_digests_ignore_bytes_and_see_values(tmp_path: Path):
    a, b = tmp_path / "a", tmp_path / "b"
    make_raw.write_outputs(tiny_interim(tmp_path), a)
    (tmp_path / "x").mkdir()
    make_raw.write_outputs(tiny_interim(tmp_path / "x"), b)
    graph_a, graph_b = a / "papkou2023_graph.graphml", b / "papkou2023_graph.graphml"
    # same values, different bytes (a comment and whitespace)
    graph_b.write_text(graph_b.read_text().replace("<graph ", "<!-- rebuilt -->\n<graph "))
    assert graph_a.read_bytes() != graph_b.read_bytes()
    assert value_digest(graph_a) == value_digest(graph_b)
    graph_b.write_text(GRAPHML.format(m="0.6"))
    assert value_digest(graph_a) != value_digest(graph_b)
    peaks = b / "papkou2023_peaks.csv"
    peaks.write_bytes(b"dna\r\nAAAAAAAAC\r\n")
    assert value_digest(a / "papkou2023_peaks.csv") == value_digest(peaks)
    peaks.write_text("dna\nAAAAAAAAA\n")
    assert value_digest(a / "papkou2023_peaks.csv") != value_digest(peaks)
    junk = tmp_path / "junk.graphml"
    junk.write_bytes(b"not xml")
    assert value_digest(junk).startswith("bytes:")


def test_report_digests_warns_without_failing(tmp_path: Path):
    out = tmp_path / "raw"
    make_raw.write_outputs(tiny_interim(tmp_path), out)
    clamped = {
        "graphml_value_sha256": value_digest(out / "papkou2023_graph.graphml"),
        "peaks_value_sha256": value_digest(out / "papkou2023_peaks.csv"),
        "graphml_bytes_sha256": "0" * 64,
        "peaks_bytes_sha256": bytes_sha256(out / "papkou2023_peaks.csv"),
    }
    log: list[str] = []
    warnings = make_raw.report_digests(out, clamped, log.append)
    assert len(warnings) == 1 and "values equal" in warnings[0]
    clamped["peaks_value_sha256"] = "1" * 64
    warnings = make_raw.report_digests(out, clamped, log.append)
    assert any("refuse" in w for w in warnings)
    assert any(line.startswith("WARNING") for line in log)


# ---- checksums, download, zip extraction --------------------------------------------


def test_md5_verification(tmp_path: Path):
    path = tmp_path / "f"
    path.write_bytes(b"hello")
    assert files.matches_md5(path, md5(b"hello"))
    assert not files.matches_md5(path, md5(b"other"))
    assert not files.matches_md5(tmp_path / "missing", md5(b"hello"))
    files.require_md5(path, md5(b"hello").upper())
    with pytest.raises(files.Md5Mismatch):
        files.require_md5(path, md5(b"other"))


def make_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)


def test_extract_members_only_named_and_verified(tmp_path: Path):
    zpath = tmp_path / "z.zip"
    make_zip(
        zpath, {"in_data/a.rds": b"AAA", "in_data/b.rds": b"BBB", "other/c.txt": b"CCC"}
    )
    root = tmp_path / "store" / "zenodo"
    status = files.extract_members(
        zpath, [("in_data/a.rds", root / "in_data/a.rds", md5(b"AAA"))]
    )
    assert list(status.values()) == ["extracted"]
    assert sorted(
        p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
    ) == ["in_data/a.rds"]
    # resume-safe: second run skips
    again = files.extract_members(
        zpath, [("in_data/a.rds", root / "in_data/a.rds", md5(b"AAA"))]
    )
    assert list(again.values()) == ["cached"]


def test_extract_members_md5_mismatch_leaves_no_file(tmp_path: Path):
    zpath = tmp_path / "z.zip"
    make_zip(zpath, {"in_data/a.rds": b"AAA"})
    dest = tmp_path / "out" / "a.rds"
    with pytest.raises(files.Md5Mismatch):
        files.extract_members(zpath, [("in_data/a.rds", dest, md5(b"nope"))])
    assert not dest.exists() and not dest.with_name("a.rds.tmp").exists()
    with pytest.raises(KeyError):
        files.extract_members(zpath, [("in_data/zzz", tmp_path / "zzz", md5(b"x"))])


def test_unpack_zip_refuses_path_escape(tmp_path: Path):
    zpath = tmp_path / "z.zip"
    make_zip(zpath, {"../evil.txt": b"x"})
    with pytest.raises(ValueError):
        files.unpack_zip(zpath, tmp_path / "dest")


class FakeResponse(io.BytesIO):
    def __init__(self, data: bytes, status: int):
        super().__init__(data)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def fake_opener(payload: bytes, ranges: bool = True):
    calls = []

    def opener(request):
        calls.append(request.headers.get("Range"))
        rng = request.headers.get("Range")
        if rng and ranges:
            start = int(rng.split("=")[1].rstrip("-"))
            return FakeResponse(payload[start:], 206)
        return FakeResponse(payload, 200)

    opener.calls = calls
    return opener


@pytest.mark.parametrize("ranges", [True, False])
def test_download_resumes_and_verifies(tmp_path: Path, ranges: bool):
    payload = b"0123456789" * 100
    dest = tmp_path / "d" / "f.bin"
    dest.parent.mkdir()
    dest.with_name("f.bin.part").write_bytes(payload[:300])
    opener = fake_opener(payload, ranges)
    assert files.download("http://x/f", dest, md5(payload), opener) == "downloaded"
    assert dest.read_bytes() == payload
    assert opener.calls == ["bytes=300-"]
    assert files.download("http://x/f", dest, md5(payload), opener) == "cached"
    assert len(opener.calls) == 1


def test_download_bad_md5_removes_partial(tmp_path: Path):
    dest = tmp_path / "f.bin"
    with pytest.raises(files.Md5Mismatch):
        files.download("http://x/f", dest, md5(b"expected"), fake_opener(b"actual"))
    assert not dest.exists() and not dest.with_name("f.bin.part").exists()


def test_download_restarts_after_a_416(tmp_path: Path):
    import urllib.error

    payload = b"0123456789" * 100
    dest = tmp_path / "f.bin"
    dest.with_name("f.bin.part").write_bytes(b"stale bytes that are not a prefix")
    inner = fake_opener(payload, ranges=True)

    def opener(request):
        if request.headers.get("Range"):
            raise urllib.error.HTTPError(request.full_url, 416, "Range", {}, None)
        return inner(request)

    assert files.download("http://x/f", dest, md5(payload), opener) == "downloaded"
    assert dest.read_bytes() == payload and not dest.with_name("f.bin.part").exists()


def test_download_accepts_a_complete_part_on_416(tmp_path: Path):
    import urllib.error

    payload = b"complete"
    dest = tmp_path / "f.bin"
    dest.with_name("f.bin.part").write_bytes(payload)

    def opener(request):
        raise urllib.error.HTTPError(request.full_url, 416, "Range", {}, None)

    assert files.download("http://x/f", dest, md5(payload), opener) == "downloaded"
    assert dest.read_bytes() == payload


def test_unpack_zip_refuses_too_many_members_and_too_many_bytes(tmp_path: Path):
    zpath = tmp_path / "z.zip"
    make_zip(zpath, {"a.txt": b"aaaa", "b.txt": b"bbbb", "c.txt": b"cccc"})
    with pytest.raises(ValueError, match="members"):
        files.unpack_zip(zpath, tmp_path / "d1", max_files=2)
    with pytest.raises(ValueError, match="bytes"):
        files.unpack_zip(zpath, tmp_path / "d2", max_bytes=10)
    assert len(files.unpack_zip(zpath, tmp_path / "d3")) == 3


@pytest.mark.parametrize("key", ["../x.csv", "a/b.csv", "..", ".hidden", "a\\b", ""])
def test_fetch_data_rejects_record_keys_that_are_not_plain_file_names(key: str):
    with pytest.raises(SystemExit, match="plain file name"):
        fetch_data.check_record_key(key)
    assert fetch_data.check_record_key("reference_data.zip") == "reference_data.zip"


def test_raw_files_are_routed_into_the_processed_raw_folder():
    destinations = files.load_config()["ours"]["destinations"]
    for name in (
        "papkou2023_fitness_wt.parquet",
        "papkou2023_fitness_diff.parquet",
        "papkou2023_graph.graphml",
        "papkou2023_peaks.csv",
    ):
        assert files.route_destination(name, destinations) == "data/processed/raw"
    assert files.route_destination("papkou2023_store.zip", destinations) == "{store}"
    assert files.route_destination("MANIFEST.tsv", destinations) == "data/processed"


def test_record_files_parses_zenodo_api():
    body = json.dumps(
        {
            "files": [
                {
                    "key": "a.zip",
                    "size": 3,
                    "checksum": "md5:abc",
                    "links": {"self": "http://u"},
                }
            ]
        }
    ).encode()
    assert files.record_files(1, lambda req: FakeResponse(body, 200)) == {
        "a.zip": files.RemoteFile("a.zip", 3, "abc", "http://u")
    }


# ---- store fetch, manifest, authors scripts, fetch-data -----------------------------


def papkou_config(zip_bytes: bytes, members: dict[str, bytes]) -> dict:
    return {
        "papkou2023": {
            "record": 1,
            "zip_name": "z.zip",
            "zip_size": len(zip_bytes),
            "zip_md5": md5(zip_bytes),
            "zip_root_in_store": "zenodo",
            "store_env": "PAPKOU2023RUGGED_DATA",
            "members": [{"in_zip": k, "md5": md5(v)} for k, v in members.items()],
            "authors_scripts": {
                "record": 2,
                "zip_name": "pipe.zip",
                "zip_size": 0,
                "zip_md5": "",
                "zip_prefix": "pipe-master/",
                "md5_file": "authors.md5",
                "store_dir": "ampseq-nextflow",
                "source_env": "X",
            },
        }
    }


def zip_bytes(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def api_for(zbytes: bytes, key: str = "pipe.zip") -> bytes:
    return json.dumps(
        {
            "files": [
                {
                    "key": key,
                    "size": len(zbytes),
                    "checksum": f"md5:{md5(zbytes)}",
                    "links": {"self": "http://zenodo/pipe.zip"},
                }
            ]
        }
    ).encode()


def test_authors_scripts_from_zenodo_zip_and_orig_backup(tmp_path: Path):
    good = {"lib/a.R": b"alpha", "scripts/b.R": b"beta"}
    zbytes = zip_bytes(
        {**{f"pipe-master/{k}": v for k, v in good.items()}, "pipe-master/junk.R": b"j"}
    )
    config = papkou_config(b"", {})
    cfg = config["papkou2023"]["authors_scripts"]
    cfg.update(zip_size=len(zbytes), zip_md5=md5(zbytes))
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "authors.md5").write_text(
        "".join(f"{md5(v)}  {k}\n" for k, v in good.items())
    )
    calls: list[str] = []

    def opener(request):
        calls.append(request.full_url)
        if request.full_url.startswith("https://zenodo.org/api/records/2"):
            return FakeResponse(api_for(zbytes), 200)
        return FakeResponse(zbytes, 200)

    store = tmp_path / "store"
    old = store / "ampseq-nextflow" / "lib" / "a.R"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"alpha ")  # differs from the pinned md5
    logs: list[str] = []
    done, missing = papkou_fetch.copy_authors_scripts(
        config, store, None, repo, opener, logs.append
    )
    assert missing == [] and len(done) == 2
    assert old.read_bytes() == b"alpha"
    assert (old.with_name("a.R.orig")).read_bytes() == b"alpha "
    assert not (store / "ampseq-nextflow" / "junk.R").exists()
    assert any("moved to ampseq-nextflow/lib/a.R.orig" in m for m in logs)
    assert any("Zenodo record 2" in m for m in logs)

    # second run: everything verified, no network, no new .orig
    calls.clear()
    papkou_fetch.copy_authors_scripts(config, store, None, repo, opener, logs.append)
    assert calls == [] and not list(old.parent.glob("a.R.orig.*"))

    # a differing file again: the earlier .orig is preserved
    old.write_bytes(b"third")
    papkou_fetch.copy_authors_scripts(config, store, None, repo, opener, logs.append)
    assert old.with_name("a.R.orig").read_bytes() == b"alpha "
    assert old.with_name("a.R.orig.1").read_bytes() == b"third"

    # a local checkout overrides the download
    old.write_bytes(b"x")
    local = tmp_path / "local"
    (local / "lib").mkdir(parents=True)
    (local / "lib" / "a.R").write_bytes(b"alpha")
    calls.clear()
    papkou_fetch.copy_authors_scripts(config, store, local, repo, opener, logs.append)
    assert calls == [] and old.read_bytes() == b"alpha"


def test_fetch_papkou_end_to_end_with_fake_network(tmp_path: Path):
    members = {"in_data/a.rds": b"AAA", "in_data/b.RDS": b"BBB"}
    zpath = tmp_path / "src.zip"
    make_zip(zpath, {**members, "in_data/big.rds": b"X" * 50})
    zbytes = zpath.read_bytes()
    config = papkou_config(zbytes, members)
    api = json.dumps(
        {
            "files": [
                {
                    "key": "z.zip",
                    "size": len(zbytes),
                    "checksum": f"md5:{md5(zbytes)}",
                    "links": {"self": "http://zenodo/z.zip"},
                }
            ]
        }
    ).encode()

    def opener(request):
        if request.full_url.startswith("https://zenodo.org/api/records/"):
            return FakeResponse(api, 200)
        return FakeResponse(zbytes, 200)

    def no_zenodo(request):
        # pipeline archive without the wanted script
        if request.full_url.startswith("https://zenodo.org/api/records/"):
            return FakeResponse(api_for(pipe_empty), 200)
        return FakeResponse(pipe_empty, 200)

    pipe_empty = zip_bytes({"pipe-master/other.R": b"o"})
    config["papkou2023"]["authors_scripts"].update(
        zip_size=len(pipe_empty), zip_md5=md5(pipe_empty)
    )
    store = tmp_path / "store"
    (store).mkdir()
    # an existing, larger manifest must survive
    (store / "MANIFEST.md5").write_text(f"{md5(b'old')} ./zenodo/in_data/old.rds\n")
    # authors' md5 list lives under repo root; use a temp repo root
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "authors.md5").write_text(f"{md5(b'script')}  lib/s.R\n")
    src = tmp_path / "ampseq"
    (src / "lib").mkdir(parents=True)
    (src / "lib" / "s.R").write_bytes(b"script")
    logs: list[str] = []
    done, missing = papkou_fetch.copy_authors_scripts(config, store, src, repo)
    assert missing == [] and done == [store / "ampseq-nextflow" / "lib" / "s.R"]
    _, missing = papkou_fetch.copy_authors_scripts(
        config, tmp_path / "empty", src / "nowhere", repo, no_zenodo, logs.append
    )
    assert missing == ["lib/s.R"]

    assert (
        papkou_fetch.fetch(
            config, store, opener=opener, log=logs.append, repo_root=repo
        )
        == 0
    )
    assert (store / "zenodo/in_data/a.rds").read_bytes() == b"AAA"
    assert not (store / "zenodo/in_data/big.rds").exists()
    manifest = files.parse_md5_lines((store / "MANIFEST.md5").read_text())
    assert (md5(b"old"), "zenodo/in_data/old.rds") in manifest
    assert (md5(b"AAA"), "zenodo/in_data/a.rds") in manifest
    assert (md5(b"script"), "ampseq-nextflow/lib/s.R") in manifest

    def no_network(_request):
        raise AssertionError("network used")

    logs.clear()
    assert (
        papkou_fetch.fetch(
            config, store, opener=no_network, log=logs.append, repo_root=repo
        )
        == 0
    )
    assert any("no download" in line for line in logs)


def test_store_path_requires_env(monkeypatch):
    monkeypatch.delenv("PAPKOU2023RUGGED_DATA", raising=False)
    with pytest.raises(SystemExit):
        papkou_fetch.store_path(files.load_config())


def test_fetch_data_empty_record_is_a_clear_error():
    config = files.load_config()
    assert config["ours"]["zenodo_record"] == ""
    with pytest.raises(SystemExit, match="zenodo_record"):
        fetch_data.fetch_data(config, store=None)


def test_fetch_data_routes_and_unpacks(tmp_path: Path):
    zpath = tmp_path / "ref.zip"
    make_zip(zpath, {"set1/a.txt": b"A"})
    zbytes = zpath.read_bytes()
    csv = b"x,y\n1,2\n"
    api = json.dumps(
        {
            "files": [
                {
                    "key": "reference_data.zip",
                    "size": len(zbytes),
                    "checksum": f"md5:{md5(zbytes)}",
                    "links": {"self": "http://z/ref"},
                },
                {
                    "key": "notes.csv",
                    "size": len(csv),
                    "checksum": f"md5:{md5(csv)}",
                    "links": {"self": "http://z/notes"},
                },
            ]
        }
    ).encode()

    def opener(request):
        url = request.full_url
        if "api/records" in url:
            return FakeResponse(api, 200)
        return FakeResponse(zbytes if url.endswith("ref") else csv, 200)

    config = {
        "ours": {
            "zenodo_record": "42",
            "destinations": [
                {"match": "reference*", "dest": "data/processed"},
                {"match": "*", "dest": "data/processed"},
            ],
        }
    }
    repo = tmp_path / "repo"
    assert (
        fetch_data.fetch_data(
            config, store=None, repo_root=repo, opener=opener, log=lambda _: None
        )
        == 0
    )
    assert (repo / "data/processed/set1/a.txt").read_bytes() == b"A"
    assert (repo / "data/processed/notes.csv").read_bytes() == csv


def test_config_and_readme_agree_on_counts():
    config = files.load_config()
    readme = (files.REPO_ROOT / "data" / "raw" / "README.md").read_text()
    for key, value in config["raw_shape"].items():
        assert f"{value:,}" in readme, key
    assert len(config["papkou2023"]["members"]) == 4
    lines = files.parse_md5_lines(
        (
            files.REPO_ROOT / config["papkou2023"]["authors_scripts"]["md5_file"]
        ).read_text()
    )
    assert len(lines) == 13
