import tensorflow as tf

from sample import top_p_logits


class TopPLogitsTest(tf.test.TestCase):
    def test_batch_of_two_uses_independent_thresholds(self):
        with self.test_session() as session:
            logits = tf.constant([
                [0.0, 3.0, 1.0, 2.0],
                [0.0, 0.5, 0.25, 0.75],
            ], dtype=tf.float32)
            filtered = top_p_logits(logits, p=0.9)
            result = session.run(filtered)

        self.assertEqual(result.shape, (2, 4))
        # The rows have different cutoffs (2.0 and 0.25).
        self.assertAllEqual(result, [
            [-1e10, 3.0, -1e10, 2.0],
            [-1e10, 0.5, 0.25, 0.75],
        ])


if __name__ == '__main__':
    tf.test.main()
