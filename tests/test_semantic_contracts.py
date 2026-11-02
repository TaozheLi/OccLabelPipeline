import importlib,json,os,sys,tempfile,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
class SemanticContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();p=Path(cls.temp.name)/'config.json';c=json.loads((ROOT/'configs/pcd.example.json').read_text());c['frame_count']=3;p.write_text(json.dumps(c));os.environ['OCC_PIPELINE_CONFIG']=str(p)
        sys.path.insert(0,str(ROOT/'occ_label_pipeline/engine'))
        cls.module=importlib.import_module('voxelize_v3_merged')
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def test_dynamic_priority_above_manual_static(self):
        static=np.full((3,3,3),14,np.uint8);manual=static.copy();dynamic=np.full_like(static,17);dynamic[1,1,1]=4
        result,manual_mask,removed,background=self.module.merge_semantics(static,manual,dynamic,True)
        self.assertEqual(result[1,1,1],4);self.assertFalse(manual_mask[1,1,1]);self.assertFalse(removed[1,1,1])
    def test_measured_return_endpoint_is_not_free(self):
        mask=self.module.free_rays(np.array([.5,.5,.5]),np.array([[2.5,.5,.5]]),np.zeros(3),np.full(3,4.),1.,(4,4,4))
        self.assertTrue(mask[0,0,0]);self.assertTrue(mask[1,0,0]);self.assertFalse(mask[2,0,0]);self.assertFalse(mask[3,0,0])
    def test_ego_never_supervised_as_free(self):
        from ego_mask import mask_arrays
        arrays={'semantics':np.full((2,2,2),17,np.uint8),'mask_lidar':np.ones((2,2,2),bool),'mask_camera':np.zeros((2,2,2),bool)}
        mask_arrays(arrays,{'bounds':[0,0,0,2,2,2],'voxel_size':1.},{'enabled':True,'bounds_grid_m':[0,0,0,1,1,1]})
        self.assertFalse(arrays['mask_lidar'][0,0,0]);self.assertTrue(arrays['mask_ego_excluded'][0,0,0]);self.assertTrue(arrays['mask_lidar'][1,1,1])
if __name__=='__main__':unittest.main()
