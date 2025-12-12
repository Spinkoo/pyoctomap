# Tree type declarations for OctoMap wrapper

cimport octomap_std
cimport octomap_math
from octomap_std cimport istream, ostream, string, list, size_t
from octomap_math cimport point3d, pose6d

cdef extern from "Pointcloud.h" namespace "octomap":
    cdef cppclass Pointcloud:
        Pointcloud() except +
        Pointcloud(const Pointcloud& other) except +
        Pointcloud(Pointcloud* other) except +
        size_t size() const
        void clear()
        void reserve(size_t size)
        void push_back(float x, float y, float z)
        void push_back(const point3d& p)
        void push_back(point3d* p)
        void push_back(const Pointcloud& other)
        void writeVrml(string filename)
        void transform(pose6d transform)
        void rotate(double roll, double pitch, double yaw)
        void transformAbsolute(pose6d transform)
        void calcBBX(point3d& lowerBound, point3d& upperBound) const
        void crop(point3d lowerBound, point3d upperBound)
        void minDist(double thres)
        void subSampleRandom(unsigned int num_samples, Pointcloud& sample_cloud)
        point3d back()
        point3d getPoint(unsigned int i) const
        const point3d& operator[](size_t i) const
        point3d& operator[](size_t i)
        istream& readBinary(istream& s)
        istream& read(istream& s)
        istream& readPCD(istream& s)
        ostream& writeBinary(ostream& s) const
        # Iterator types
        cppclass iterator:
            iterator() except +
            point3d& operator*()
            iterator& operator++()
            bool operator==(iterator& other)
            bool operator!=(iterator& other)
        cppclass const_iterator:
            const_iterator() except +
            const point3d& operator*()
            const_iterator& operator++()
            bool operator==(const_iterator& other)
            bool operator!=(const_iterator& other)
        iterator begin()
        iterator end()
        const_iterator begin() const
        const_iterator end() const

cdef extern from "OcTreeNode.h" namespace "octomap":
    cdef cppclass OcTreeNode:
        OcTreeNode() except +
        void addValue(float& p)
        bool childExists(unsigned int i)
        float getValue()
        void setValue(float v)
        double getOccupancy()
        OcTreeNode* getChild(unsigned int i)
        float getLogOdds()
        void setLogOdds(float l)
        bool hasChildren()  # Deprecated - use tree.nodeHasChildren(node) instead
        float getMaxChildLogOdds()
        void updateOccupancyChildren()

cdef extern from "OcTreeKey.h" namespace "octomap":
    ctypedef unsigned short int key_type
    cdef struct OcTreeKey:
        key_type k[3]
    OcTreeKey computeIndexKey(key_type level, const OcTreeKey& key)
    unsigned int computeChildIdx(const OcTreeKey& key, int depth)
    
    cdef cppclass KeyRay:
        KeyRay() except +
        void reset()
        void addKey(const OcTreeKey& k)
        size_t size() const
        
        cppclass iterator:
            iterator() except +
            OcTreeKey& deref()
            OcTreeKey& operator*()
            iterator& inc()
            iterator& operator++()
            bool operator!=(iterator& other)
        
        iterator begin()
        iterator end()

cdef extern from "include_and_setting.h" namespace "octomap":
    cdef cppclass OccupancyOcTreeBase[T]:
        cppclass iterator_base:
            point3d getCoordinate()
            unsigned int getDepth()
            OcTreeKey getIndexKey()
            OcTreeKey& getKey()
            double getSize() except +
            double getX() except +
            double getY() except +
            double getZ() except +
            OcTreeNode& operator*()
            bool operator==(iterator_base &other)
            bool operator!=(iterator_base &other)

        cppclass tree_iterator(iterator_base):
            tree_iterator() except +
            tree_iterator(tree_iterator&) except +
            tree_iterator& operator++()
            bool operator==(tree_iterator &other)
            bool operator!=(tree_iterator &other)
            bool isLeaf() except +

        cppclass leaf_iterator(iterator_base):
            leaf_iterator() except +
            leaf_iterator(leaf_iterator&) except +
            leaf_iterator& operator++()
            bool operator==(leaf_iterator &other)
            bool operator!=(leaf_iterator &other)

        cppclass leaf_bbx_iterator(iterator_base):
            leaf_bbx_iterator() except +
            leaf_bbx_iterator(leaf_bbx_iterator&) except +
            leaf_bbx_iterator& operator++()
            bool operator==(leaf_bbx_iterator &other)
            bool operator!=(leaf_bbx_iterator &other)

cdef extern from "include_and_setting.h" namespace "octomap":
    cdef cppclass OcTree:
        OcTree(double resolution) except +
        OcTree(string _filename) except +
        OcTreeKey adjustKeyAtDepth(OcTreeKey& key, unsigned int depth)
        unsigned short int adjustKeyAtDepth(unsigned short int key, unsigned int depth)
        bool bbxSet()
        size_t calcNumNodes()
        void clear()
        OcTreeKey coordToKey(point3d& coord)
        OcTreeKey coordToKey(point3d& coord, unsigned int depth)
        bool coordToKeyChecked(point3d& coord, OcTreeKey& key)
        bool coordToKeyChecked(point3d& coord, unsigned int depth, OcTreeKey& key)
        bool deleteNode(point3d& value, unsigned int depth)
        bool castRay(point3d& origin, point3d& direction, point3d& end,
                     bool ignoreUnknownCells, double maxRange) except +
        bool computeRayKeys(point3d& origin, point3d& end, KeyRay& ray)
        OcTree* read(string& filename)
        OcTree* read(istream& s)
        bool write(string& filename)
        bool write(ostream& s)
        bool readBinary(string& filename)
        bool readBinary(istream& s)
        bool writeBinary(string& filename)
        bool writeBinary(ostream& s)
        bool isNodeOccupied(OcTreeNode& occupancyNode)
        bool isNodeAtThreshold(OcTreeNode& occupancyNode)
        void insertPointCloud(Pointcloud& scan, point3d& sensor_origin,
                              double maxrange, bool lazy_eval, bool discretize)
        void insertPointCloudRays(const Pointcloud& scan, const point3d& sensor_origin, double maxrange, bool lazy_eval)
        OccupancyOcTreeBase[OcTreeNode].tree_iterator begin_tree(unsigned char maxDepth) except +
        OccupancyOcTreeBase[OcTreeNode].tree_iterator end_tree() except +
        OccupancyOcTreeBase[OcTreeNode].leaf_iterator begin_leafs(unsigned char maxDepth) except +
        OccupancyOcTreeBase[OcTreeNode].leaf_iterator end_leafs() except +
        OccupancyOcTreeBase[OcTreeNode].leaf_bbx_iterator begin_leafs_bbx(point3d &min, point3d &max, unsigned char maxDepth) except +
        OccupancyOcTreeBase[OcTreeNode].leaf_bbx_iterator end_leafs_bbx() except +
        point3d getBBXBounds()
        point3d getBBXCenter()
        point3d getBBXMax()
        point3d getBBXMin()
        OcTreeNode* getRoot()
        size_t getNumLeafNodes()
        double getResolution()
        unsigned int getTreeDepth()
        string getTreeType()
        bool inBBX(point3d& p)
        point3d keyToCoord(OcTreeKey& key)
        point3d keyToCoord(OcTreeKey& key, unsigned int depth)
        unsigned long long memoryFullGrid()
        size_t memoryUsage()
        size_t memoryUsageNode()
        void resetChangeDetection()
        OcTreeNode* search(double x, double y, double z, unsigned int depth)
        OcTreeNode* search(point3d& value, unsigned int depth)
        OcTreeNode* search(OcTreeKey& key, unsigned int depth)
        void setBBXMax(point3d& max)
        void setBBXMin(point3d& min)
        void setResolution(double r)
        size_t size()
        void toMaxLikelihood()
        OcTreeNode* updateNode(double x, double y, double z, float log_odds_update, bool lazy_eval)
        OcTreeNode* updateNode(double x, double y, double z, bool occupied, bool lazy_eval)
        OcTreeNode* updateNode(OcTreeKey& key, float log_odds_update, bool lazy_eval)
        OcTreeNode* updateNode(OcTreeKey& key, bool occupied, bool lazy_eval)
        bool computeRayKeys(const point3d& origin, const point3d& end, KeyRay& ray)
        void updateInnerOccupancy()
        void useBBXLimit(bool enable)
        double volume()

        double getClampingThresMax()
        float getClampingThresMaxLog()
        double getClampingThresMin()
        float getClampingThresMinLog()

        double getOccupancyThres()
        float getOccupancyThresLog()
        double getProbHit()
        float getProbHitLog()
        double getProbMiss()
        float getProbMissLog()

        void setClampingThresMax(double thresProb)
        void setClampingThresMin(double thresProb)
        void setOccupancyThres(double prob)
        void setProbHit(double prob)
        void setProbMiss(double prob)

        void getMetricSize(double& x, double& y, double& z)
        void getMetricMin(double& x, double& y, double& z)
        void getMetricMax(double& x, double& y, double& z)

        void expandNode(OcTreeNode* node)
        OcTreeNode* createNodeChild(OcTreeNode *node, unsigned int childIdx)
        OcTreeNode* getNodeChild(OcTreeNode *node, unsigned int childIdx)
        bool isNodeCollapsible(const OcTreeNode* node)
        void deleteNodeChild(OcTreeNode *node, unsigned int childIdx)
        bool pruneNode(OcTreeNode *node)
        bool nodeHasChildren(const OcTreeNode* node)
        void prune()
        void expand()

cdef extern from "ColorOcTree.h" namespace "octomap":
    cdef cppclass ColorOcTreeNode(OcTreeNode):
        cppclass Color:
            Color() except +
            Color(unsigned char r, unsigned char g, unsigned char b) except +
            unsigned char r
            unsigned char g
            unsigned char b
            bool operator==(const Color& other)
            bool operator!=(const Color& other)
        
        ColorOcTreeNode() except +
        ColorOcTreeNode(const ColorOcTreeNode& rhs) except +
        Color getColor() const
        Color& getColor()
        void setColor(Color c)
        void setColor(unsigned char r, unsigned char g, unsigned char b)
        bool isColorSet() const
        void updateColorChildren()
        Color getAverageChildColor() const
        void copyData(const ColorOcTreeNode& other)
        bool operator==(const ColorOcTreeNode& rhs) const
    
    cdef cppclass ColorOcTree:
        ColorOcTree(double resolution) except +
        ColorOcTree* create() const
        string getTreeType() const
        bool pruneNode(ColorOcTreeNode* node)
        bool isNodeCollapsible(const ColorOcTreeNode* node) const
        ColorOcTreeNode* setNodeColor(const OcTreeKey& key, unsigned char r, unsigned char g, unsigned char b)
        ColorOcTreeNode* setNodeColor(float x, float y, float z, unsigned char r, unsigned char g, unsigned char b)
        ColorOcTreeNode* averageNodeColor(const OcTreeKey& key, unsigned char r, unsigned char g, unsigned char b)
        ColorOcTreeNode* averageNodeColor(float x, float y, float z, unsigned char r, unsigned char g, unsigned char b)
        ColorOcTreeNode* integrateNodeColor(const OcTreeKey& key, unsigned char r, unsigned char g, unsigned char b)
        ColorOcTreeNode* integrateNodeColor(float x, float y, float z, unsigned char r, unsigned char g, unsigned char b)
        void updateInnerOccupancy()
        void writeColorHistogram(string filename)
        # Inherited from OccupancyOcTreeBase - need to declare key methods
        OcTreeKey coordToKey(point3d& coord)
        OcTreeKey coordToKey(point3d& coord, unsigned int depth)
        bool coordToKeyChecked(point3d& coord, OcTreeKey& key)
        bool coordToKeyChecked(point3d& coord, unsigned int depth, OcTreeKey& key)
        ColorOcTreeNode* search(double x, double y, double z, unsigned int depth)
        ColorOcTreeNode* search(point3d& value, unsigned int depth)
        ColorOcTreeNode* search(const OcTreeKey& key, unsigned int depth)
        ColorOcTreeNode* updateNode(double x, double y, double z, float log_odds_update, bool lazy_eval)
        ColorOcTreeNode* updateNode(double x, double y, double z, bool occupied, bool lazy_eval)
        ColorOcTreeNode* updateNode(const OcTreeKey& key, float log_odds_update, bool lazy_eval)
        ColorOcTreeNode* updateNode(const OcTreeKey& key, bool occupied, bool lazy_eval)
        void insertPointCloud(Pointcloud& scan, point3d& sensor_origin, double maxrange, bool lazy_eval, bool discretize)
        bool isNodeOccupied(ColorOcTreeNode& occupancyNode)
        bool isNodeAtThreshold(ColorOcTreeNode& occupancyNode)
        bool castRay(point3d& origin, point3d& direction, point3d& end, bool ignoreUnknownCells, double maxRange) except +
        double getResolution()
        unsigned int getTreeDepth()
        size_t size()
        size_t getNumLeafNodes()
        size_t calcNumNodes()
        void clear()
        bool write(string& filename)
        bool write(ostream& s)
        bool readBinary(string& filename)
        bool writeBinary(string& filename)
        ColorOcTree* read(string& filename)
        ColorOcTree* read(istream& s)
        point3d getBBXMin()
        point3d getBBXMax()
        point3d getBBXCenter()
        point3d getBBXBounds()
        void setBBXMin(point3d& min)
        void setBBXMax(point3d& max)
        bool inBBX(point3d& p)
        point3d keyToCoord(OcTreeKey& key)
        point3d keyToCoord(OcTreeKey& key, unsigned int depth)
        ColorOcTreeNode* getRoot()
        bool nodeHasChildren(const ColorOcTreeNode* node)
        ColorOcTreeNode* getNodeChild(ColorOcTreeNode *node, unsigned int childIdx)
        ColorOcTreeNode* createNodeChild(ColorOcTreeNode *node, unsigned int childIdx)
        void deleteNodeChild(ColorOcTreeNode *node, unsigned int childIdx)
        void expandNode(ColorOcTreeNode* node)
        OccupancyOcTreeBase[ColorOcTreeNode].tree_iterator begin_tree(unsigned char maxDepth) except +
        OccupancyOcTreeBase[ColorOcTreeNode].tree_iterator end_tree() except +
        OccupancyOcTreeBase[ColorOcTreeNode].leaf_iterator begin_leafs(unsigned char maxDepth) except +
        OccupancyOcTreeBase[ColorOcTreeNode].leaf_iterator end_leafs() except +
        OccupancyOcTreeBase[ColorOcTreeNode].leaf_bbx_iterator begin_leafs_bbx(point3d &min, point3d &max, unsigned char maxDepth) except +
        OccupancyOcTreeBase[ColorOcTreeNode].leaf_bbx_iterator end_leafs_bbx() except +
        void getMetricSize(double& x, double& y, double& z)
        void getMetricMin(double& x, double& y, double& z)
        void getMetricMax(double& x, double& y, double& z)
        size_t memoryUsage()
        size_t memoryUsageNode()
        double volume()
        void prune()
        void expand()
        void useBBXLimit(bool enable)
        double getOccupancyThres()
        float getOccupancyThresLog()
        double getProbHit()
        float getProbHitLog()
        double getProbMiss()
        float getProbMissLog()
        void setOccupancyThres(double prob)
        void setProbHit(double prob)
        void setProbMiss(double prob)
        double getClampingThresMax()
        float getClampingThresMaxLog()
        double getClampingThresMin()
        float getClampingThresMinLog()
        void setClampingThresMax(double thresProb)
        void setClampingThresMin(double thresProb)

cdef extern from "CountingOcTree.h" namespace "octomap":
    cdef cppclass CountingOcTreeNode:
        CountingOcTreeNode() except +
        unsigned int getCount() const
        void increaseCount()
        void setCount(unsigned int c)
        unsigned int getValue() const
        void setValue(unsigned int v)
    
    cdef cppclass CountingOcTree:
        CountingOcTree(double resolution) except +
        CountingOcTree* create() const
        string getTreeType() const
        CountingOcTreeNode* updateNode(const point3d& value)
        CountingOcTreeNode* updateNode(const OcTreeKey& k)
        void getCentersMinHits(list[point3d]& node_centers, unsigned int min_hits) const
        # Inherited from OcTreeBase - need to declare key methods
        OcTreeKey coordToKey(point3d& coord)
        OcTreeKey coordToKey(point3d& coord, unsigned int depth)
        bool coordToKeyChecked(point3d& coord, OcTreeKey& key)
        bool coordToKeyChecked(point3d& coord, unsigned int depth, OcTreeKey& key)
        CountingOcTreeNode* search(double x, double y, double z, unsigned int depth)
        CountingOcTreeNode* search(point3d& value, unsigned int depth)
        CountingOcTreeNode* search(const OcTreeKey& key, unsigned int depth)
        double getResolution()
        unsigned int getTreeDepth()
        size_t size()
        size_t getNumLeafNodes()
        size_t calcNumNodes()
        void clear()
        bool write(string& filename)
        bool write(ostream& s)
        CountingOcTree* read(string& filename)
        CountingOcTree* read(istream& s)
        point3d keyToCoord(OcTreeKey& key)
        point3d keyToCoord(OcTreeKey& key, unsigned int depth)
        CountingOcTreeNode* getRoot()
        bool nodeHasChildren(const CountingOcTreeNode* node)
        CountingOcTreeNode* getNodeChild(CountingOcTreeNode *node, unsigned int childIdx)
        CountingOcTreeNode* createNodeChild(CountingOcTreeNode *node, unsigned int childIdx)
        void deleteNodeChild(CountingOcTreeNode *node, unsigned int childIdx)
        void expandNode(CountingOcTreeNode* node)
        bool isNodeCollapsible(const CountingOcTreeNode* node) const
        bool pruneNode(CountingOcTreeNode* node)
        void getMetricSize(double& x, double& y, double& z)
        void getMetricMin(double& x, double& y, double& z)
        void getMetricMax(double& x, double& y, double& z)
        size_t memoryUsage()
        size_t memoryUsageNode()
        double volume()

cdef extern from "OcTreeStamped.h" namespace "octomap":
    cdef cppclass OcTreeNodeStamped(OcTreeNode):
        OcTreeNodeStamped() except +
        OcTreeNodeStamped(const OcTreeNodeStamped& rhs) except +
        unsigned int getTimestamp() const
        void updateTimestamp()
        void setTimestamp(unsigned int t)
        void updateOccupancyChildren()
        void copyData(const OcTreeNodeStamped& other)
        bool operator==(const OcTreeNodeStamped& rhs) const
    
    cdef cppclass OcTreeStamped:
        OcTreeStamped(double resolution) except +
        OcTreeStamped* create() const
        string getTreeType() const
        unsigned int getLastUpdateTime()
        void degradeOutdatedNodes(unsigned int time_thres)
        void updateNodeLogOdds(OcTreeNodeStamped* node, const float& update) const
        void integrateMissNoTime(OcTreeNodeStamped* node) const
        # Inherited from OccupancyOcTreeBase - updateNode methods
        OcTreeNodeStamped* updateNode(const OcTreeKey& key, float log_odds_update, bool lazy_eval)
        OcTreeNodeStamped* updateNode(point3d& value, float log_odds_update, bool lazy_eval)
        OcTreeNodeStamped* updateNode(double x, double y, double z, float log_odds_update, bool lazy_eval)
        OcTreeNodeStamped* updateNode(const OcTreeKey& key, bool occupied, bool lazy_eval)
        OcTreeNodeStamped* updateNode(point3d& value, bool occupied, bool lazy_eval)
        OcTreeNodeStamped* updateNode(double x, double y, double z, bool occupied, bool lazy_eval)
        # Inherited from OccupancyOcTreeBase - need to declare key methods
        OcTreeKey coordToKey(point3d& coord)
        OcTreeKey coordToKey(point3d& coord, unsigned int depth)
        bool coordToKeyChecked(point3d& coord, OcTreeKey& key)
        bool coordToKeyChecked(point3d& coord, unsigned int depth, OcTreeKey& key)
        OcTreeNodeStamped* search(double x, double y, double z, unsigned int depth)
        OcTreeNodeStamped* search(point3d& value, unsigned int depth)
        OcTreeNodeStamped* search(const OcTreeKey& key, unsigned int depth)
        double getResolution()
        unsigned int getTreeDepth()
        size_t size()
        size_t getNumLeafNodes()
        size_t calcNumNodes()
        void clear()
        bool write(string& filename)
        bool write(ostream& s)
        OcTreeStamped* read(string& filename)
        OcTreeStamped* read(istream& s)
        bool readBinary(string& filename)
        bool readBinary(istream& s)
        bool writeBinary(string& filename)
        bool writeBinary(ostream& s)
        bool isNodeOccupied(OcTreeNodeStamped* node)
        bool isNodeAtThreshold(OcTreeNodeStamped* node)
        bool castRay(point3d& origin, point3d& direction, point3d& end,
                     bool ignoreUnknownCells, double maxRange) except +
        point3d keyToCoord(OcTreeKey& key)
        point3d keyToCoord(OcTreeKey& key, unsigned int depth)
        OcTreeNodeStamped* getRoot()
        bool nodeHasChildren(const OcTreeNodeStamped* node)
        OcTreeNodeStamped* getNodeChild(OcTreeNodeStamped *node, unsigned int childIdx)
        OcTreeNodeStamped* createNodeChild(OcTreeNodeStamped *node, unsigned int childIdx)
        void deleteNodeChild(OcTreeNodeStamped *node, unsigned int childIdx)
        void expandNode(OcTreeNodeStamped* node)
        bool isNodeCollapsible(const OcTreeNodeStamped* node) const
        void updateInnerOccupancy()
        bool pruneNode(OcTreeNodeStamped* node)
        void getMetricSize(double& x, double& y, double& z)
        void getMetricMin(double& x, double& y, double& z)
        void getMetricMax(double& x, double& y, double& z)
        size_t memoryUsage()
        size_t memoryUsageNode()
        double volume()
        point3d getBBXMin()
        point3d getBBXMax()
        point3d getBBXCenter()
        point3d getBBXBounds()
        void setBBXMin(point3d& value)
        void setBBXMax(point3d& value)
        bool inBBX(point3d& value)
        void insertPointCloud(Pointcloud& scan, point3d& sensor_origin,
                              double maxrange, bool lazy_eval, bool discretize)
        OccupancyOcTreeBase[OcTreeNodeStamped].tree_iterator begin_tree(unsigned char maxDepth) except +
        OccupancyOcTreeBase[OcTreeNodeStamped].tree_iterator end_tree() except +
        OccupancyOcTreeBase[OcTreeNodeStamped].leaf_iterator begin_leafs(unsigned char maxDepth) except +
        OccupancyOcTreeBase[OcTreeNodeStamped].leaf_iterator end_leafs() except +
        OccupancyOcTreeBase[OcTreeNodeStamped].leaf_bbx_iterator begin_leafs_bbx(point3d &min, point3d &max, unsigned char maxDepth) except +
        OccupancyOcTreeBase[OcTreeNodeStamped].leaf_bbx_iterator end_leafs_bbx() except +

