import ast,json,tempfile,unittest
from pathlib import Path
import numpy as np
from occ_label_pipeline.config import load_config
from occ_label_pipeline.prepare import rotate_xyz,write_pcd
ROOT=Path(__file__).resolve().parents[1]
class PortabilityTests(unittest.TestCase):
    def test_vehicle_rotation_preserves_z_and_distances(self):
        x=np.array([[1.,2.,-1.75],[-3.,4.,1.]])
        y=rotate_xyz(x,180)
        np.testing.assert_allclose(y,x*np.array([-1,-1,1]),atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(y,axis=1),np.linalg.norm(x,axis=1),rtol=1e-6)
    def test_relative_config_paths_are_config_relative(self):
        c,p=load_config(ROOT/'configs/pcd.example.json')
        self.assertEqual(Path(c['output_root']),ROOT/'outputs/example_capture')
        self.assertNotIn('/home/mila',(ROOT/'configs/pcd.example.json').read_text())
    def test_input_output_alias_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            c=json.loads((ROOT/'configs/pcd.example.json').read_text());c['output_root']=c['data_root']='same';p=Path(t)/'c.json';p.write_text(json.dumps(c))
            with self.assertRaises(ValueError):load_config(p)
    def test_binary_pcd_keeps_measured_count(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'p.pcd';x=np.array([[1,2,3],[4,5,6]],np.float32);write_pcd(p,x,np.array([100,255]),np.array([100,1000]))
            raw=p.read_bytes();header,data=raw.split(b'DATA binary\n');self.assertIn(b'POINTS 2',header);self.assertEqual(len(data),40)
            rows=np.frombuffer(data,np.float32).reshape(2,5);np.testing.assert_array_equal(rows[:,:3],x);np.testing.assert_array_equal(rows[:,4],[100,1000])
    def test_python_sources_compile_and_no_private_paths(self):
        for p in (ROOT/'occ_label_pipeline').rglob('*.py'):
            s=p.read_text();ast.parse(s);self.assertNotIn('/home/mila/',s,str(p));self.assertNotIn('20261012_142203',s,str(p))
if __name__=='__main__':unittest.main()
