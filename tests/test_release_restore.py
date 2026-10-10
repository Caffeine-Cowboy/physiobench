"""Publication transport safety; no research source or result modification."""
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import unittest
from unittest.mock import patch
import tempfile
import zipfile


SPEC = importlib.util.spec_from_file_location("restore_release", Path(__file__).resolve().parents[1]/"scripts"/"restore_release_artifacts.py")
restorer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(restorer)


class RestorationSafety(unittest.TestCase):
    def fixture(self, directory, extra=None):
        root = Path(directory)
        (root/'release').mkdir()
        path = 'results/neural-study/checkpoints/synthetic-test.pt'
        data = b'fixed independent synthetic test bytes'
        with zipfile.ZipFile(root/restorer.BUNDLE_NAME,'w') as archive:
            archive.writestr('physiobench/'+path,data)
            archive.writestr('physiobench/src/physiobench/simulator.py',b'archive source must never replace checkout')
            if extra is not None: archive.writestr(extra[0],extra[1])
        checksum = restorer.digest(root/restorer.BUNDLE_NAME)
        manifest = {'schema_version':1,'bundle':restorer.BUNDLE_NAME,'bundle_sha256':checksum,
                    'archive_prefix':restorer.ARCHIVE_PREFIX,'n_synthetic_artifacts':1,
                    'artifacts_sha256':{path:hashlib.sha256(data).hexdigest()}}
        (root/restorer.MANIFEST_NAME).write_text(json.dumps(manifest))
        return root,path,data,checksum

    def execute(self, root,checksum):
        with patch.object(restorer,'BUNDLE_SHA256',checksum),patch.object(restorer,'EXPECTED_ARTIFACT_COUNT',1):
            return restorer.restore(root)

    def test_restoration_is_idempotent_and_never_writes_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root,path,data,checksum=self.fixture(directory)
            source=root/'src/physiobench/simulator.py'
            source.parent.mkdir(parents=True);source.write_text('existing checkout source')
            self.assertEqual(self.execute(root,checksum)['restored_files'],1)
            self.assertEqual((root/path).read_bytes(),data)
            self.assertEqual(source.read_text(),'existing checkout source')
            self.assertEqual(self.execute(root,checksum)['already_matching_files'],1)

    def test_existing_conflict_is_refused_before_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root,path,_,checksum=self.fixture(directory)
            destination=root/path;destination.parent.mkdir(parents=True);destination.write_bytes(b'user work')
            with self.assertRaises(FileExistsError):self.execute(root,checksum)
            self.assertEqual(destination.read_bytes(),b'user work')

    def test_tampered_bundle_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root,_,_,checksum=self.fixture(directory)
            with (root/restorer.BUNDLE_NAME).open('ab') as handle:handle.write(b'tamper')
            with self.assertRaises(ValueError):self.execute(root,checksum)
            self.assertFalse((root/'results').exists())

    def test_archive_traversal_is_refused_before_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root,_,_,checksum=self.fixture(directory,('../escaped.txt',b'bad'))
            with self.assertRaises(ValueError):self.execute(root,checksum)
            self.assertFalse((root/'results').exists())

    def test_archive_symlink_is_refused_before_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            entry=zipfile.ZipInfo('physiobench/link')
            entry.create_system=3;entry.external_attr=(stat.S_IFLNK|0o777)<<16
            root,_,_,checksum=self.fixture(directory,(entry,b'/outside'))
            with self.assertRaises(ValueError):self.execute(root,checksum)
            self.assertFalse((root/'results').exists())

    def test_destination_symlink_is_refused(self):
        with tempfile.TemporaryDirectory() as directory,tempfile.TemporaryDirectory() as outside:
            root,_,_,checksum=self.fixture(directory)
            (root/'results').symlink_to(outside,target_is_directory=True)
            with self.assertRaises(ValueError):self.execute(root,checksum)
            self.assertEqual(list(Path(outside).iterdir()),[])

    def test_duplicate_archive_member_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root,path,data,checksum=self.fixture(directory,('physiobench/results/neural-study/checkpoints/synthetic-test.pt',b'duplicate'))
            with self.assertRaises(ValueError):self.execute(root,checksum)
            self.assertFalse((root/'results').exists())

    def test_manifest_cannot_target_source_even_with_expected_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root,path,data,checksum=self.fixture(directory)
            value=json.loads((root/restorer.MANIFEST_NAME).read_text())
            value['artifacts_sha256']={'src/physiobench/simulator.py':hashlib.sha256(data).hexdigest()}
            (root/restorer.MANIFEST_NAME).write_text(json.dumps(value))
            with self.assertRaises(ValueError):self.execute(root,checksum)
            self.assertFalse((root/'src').exists())


if __name__=='__main__':
    unittest.main()
