import multer from 'multer';
import nextConnect from 'next-connect';
import { NextApiRequest, NextApiResponse } from 'next';

type MulterRequest = NextApiRequest & {
  file: Express.Multer.File;
};

const upload = multer({ dest: 'public/uploads/' });

const apiRoute = nextConnect({
  onError(err, req, res: NextApiResponse) {
    res.status(500).end(`Error: ${err.message}`);
  },
});

apiRoute.use(upload.single('video'));

apiRoute.post((req: MulterRequest, res: NextApiResponse) => {
  res.status(200).json({ filePath: `/uploads/${req.file.filename}` });
});

export default apiRoute;

export const config = {
  api: { bodyParser: false },
};
